"""Let keyframes and refs coexist, and give per-block strength."""

import logging
import torch

import comfy.ldm.common_dit
import comfy.ldm.minimax.model as mm_model
import comfy.model_base as model_base

from .patch_layout import (
    MC_AUDIO_STRENGTH,
    MC_VIDEO_STRENGTH,
)

_LOG = logging.getLogger("h3_motion_context")
_ORIG = {}


def _mix(r, aug, seed):
    """Stock global-aug row mixing for unmarked blocks."""
    if aug < 1.0:
        gen = torch.Generator("cpu").manual_seed(seed)
        noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)
        r = aug * r + (1.0 - aug) * noise.to(r.device)
    return r


def _install(cls, attr, patched, gone_msg, done_msg):
    """Adopt or install `patched` over `cls.attr`; returns success."""
    if cls is None or not hasattr(cls, attr):
        _LOG.warning("h3_motion_context: %s not found, %s", attr, gone_msg)
        return False
    current = getattr(cls, attr)
    if getattr(current, "_h3mc_patcher", False):
        _ORIG[attr] = getattr(current, "_h3mc_orig", current)
        return True
    _ORIG[attr] = current
    setattr(patched, "_h3mc_patcher", True)
    setattr(patched, "_h3mc_orig", current)
    setattr(cls, attr, patched)
    _LOG.info("h3_motion_context: %s", done_msg)
    return True


def _patched_extra_conds(self, **kwargs):
    out = _ORIG["extra_conds"](self, **kwargs)

    cond = out.get("minimax_payload", None)
    payload = getattr(cond, "cond", None) if cond is not None else None
    if not isinstance(payload, dict):
        _LOG.warning("h3_motion_context: could not reach the H3 payload, "
                     "keyframe latents may have been overwritten by refs")
        return out

    hard = kwargs.get("minimax_hard_video", None)
    if hard is not None:
        payload["minimax_hard_video"] = hard
        _LOG.info("h3_motion_context: %d hard-injected video steps passed "
                  "to the sampling payload", len(hard))

    keyframes = kwargs.get("minimax_keyframes", None)
    refs = kwargs.get("minimax_refs", None)
    if not keyframes or not refs:
        return out

    kf_video = [kf["latent"] for kf in keyframes if "latent" in kf]
    ref_video = [r["latent"] for r in refs if "latent" in r]
    payload["cond_video_latents"] = kf_video + ref_video
    payload["cond_audio_latents"] = [r["audio_latent"] for r in refs
                                     if r.get("audio_latent") is not None]

    fc = kwargs.get("minimax_frame_count", None)
    if fc is not None:
        payload["frame_count"] = fc
    return out


def apply_patch():
    return _install(
        getattr(model_base, "MiniMaxH3", None), "extra_conds",
        _patched_extra_conds,
        "keyframes and refs cannot be combined",
        "keyframe/ref coexistence enabled")


def _patched_cond_audio_rows(self, payload, device):
    active = payload.get("_h3mc_active_refs")
    rows = []
    aug = float(payload.get("audio_cond_noise_aug", mm_model.AUDIO_COND_TIMESTEP))
    seed = int(payload.get("seed", 0)) + 1
    for r in payload.get("refs") or []:
        if r.get("audio_latent") is None:
            continue
        if active is not None and not any(r is x for x in active):
            continue
        z = r["audio_latent"]
        rr = mm_model.pack_audio(z.to(torch.float32))
        if r.get(MC_AUDIO_STRENGTH) is None:
            rr = _mix(rr, aug, seed)
        rows.append(rr.to(device))
    return torch.cat(rows, dim=0) if rows else None


def apply_cond_audio_patch():
    return _install(
        getattr(mm_model, "MiniMaxH3Model", None), "_cond_audio_rows",
        _patched_cond_audio_rows,
        "per-block audio strength unavailable",
        "per-block audio strength enabled")


def _patched_forward(self, x, timestep, context, transformer_options={},
                     minimax_payload=None, **kwargs):
    payload = minimax_payload or {}
    kfs = payload.get("keyframes") or []
    refs = payload.get("refs") or []
    weak_v = [kf for kf in kfs
              if kf.get(MC_VIDEO_STRENGTH) is not None
              and float(kf[MC_VIDEO_STRENGTH]) < 1.0]
    weak_a = [r for r in refs
              if r.get(MC_AUDIO_STRENGTH) is not None
              and float(r[MC_AUDIO_STRENGTH]) < 1.0]
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    shift_v = transformer_options.get("minimax_h3_sigma_shift_video")
    if shift_v is None:
        shift_v = self.sigma_shift_video
    shift_a = transformer_options.get("minimax_h3_sigma_shift_audio")
    if shift_a is None:
        shift_a = self.sigma_shift_audio
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - mm_model.time_shift_sigma(sigma_v, float(shift_v),
                                                float(shift_a)))
    if weak_v:
        payload["visual_cond_noise_aug"] = 0.999
    if weak_a:
        payload["audio_cond_noise_aug"] = 0.999

    keep_kf = [kf for kf in kfs
               if not (kf.get(MC_VIDEO_STRENGTH) is not None
                       and float(kf[MC_VIDEO_STRENGTH]) < 1.0
                       and t_v >= float(kf[MC_VIDEO_STRENGTH]))]
    keep_refs = [r for r in refs
                 if not (r.get(MC_AUDIO_STRENGTH) is not None
                         and float(r[MC_AUDIO_STRENGTH]) < 1.0
                         and t_a >= float(r[MC_AUDIO_STRENGTH]))]
    if len(keep_kf) != len(kfs) or len(keep_refs) != len(refs):
        mark = payload.get("_h3mc_active_keyframes")
        mark_a = payload.get("_h3mc_active_refs")
        same = (mark is not None and [id(k) for k in mark] == [id(k) for k in keep_kf]
                and mark_a is not None
                and [id(r) for r in mark_a] == [id(r) for r in keep_refs])
        if not same:
            video_x = comfy.ldm.common_dit.pad_to_patch_size(x[0], self.patch_size)
            pl_kw = {"keyframes": keep_kf, "refs": keep_refs}
            fc = payload.get("frame_count")
            if fc is not None and "frame_count" in getattr(mm_model.PackedLayout.__init__, "__code__", object()).co_varnames:
                pl_kw["frame_count"] = fc
            payload["layout"] = mm_model.PackedLayout(
                context.shape[1], video_x.shape[2], video_x.shape[3],
                video_x.shape[4], x[1].shape[-1], **pl_kw)
            payload["_h3mc_active_keyframes"] = keep_kf
            payload["_h3mc_active_refs"] = keep_refs
    out = _ORIG["forward"](self, x, timestep, context,
                           transformer_options, minimax_payload, **kwargs)
    _clamp_hard(out, x, timestep, payload)
    return out


def _clamp_hard(out, x, timestep, payload):
    hard = payload.get("minimax_hard_video")
    if not hard or not out or out[0] is None or x[0].ndim != 5:
        return
    sigma = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    v = out[0]
    applied = []
    for entry in hard:
        t = int(entry["index"])
        if t >= v.shape[2]:
            continue
        lat = entry["latent"].to(device=v.device, dtype=v.dtype)
        v[:, :, t:t + 1] = (x[0][:, :, t:t + 1] - lat) / sigma
        applied.append(t)
    if applied:
        _LOG.info("h3_motion_context: clamped %d hard video steps at sigma "
                  "%.4f: %s", len(applied), float(sigma), applied)


def apply_forward_patch():
    return _install(
        getattr(mm_model, "MiniMaxH3Model", None), "forward",
        _patched_forward,
        "per-keyframe video strength unavailable",
        "per-keyframe video strength enabled")


def _patched_cond_video_rows(self, payload, device):
    active_kf = payload.get("_h3mc_active_keyframes")
    active_refs = payload.get("_h3mc_active_refs")
    if active_kf is None and active_refs is None:
        return _ORIG["_cond_video_rows"](self, payload, device)
    rows = []
    aug = float(payload.get("visual_cond_noise_aug",
                            mm_model.VISUAL_COND_TIMESTEP))
    seed = int(payload.get("seed", 0))
    def _add_row(item):
        r = mm_model.patchify_video(item["latent"].to(torch.float32),
                                    self.patch_size)
        if item.get(MC_VIDEO_STRENGTH) is None:
            r = _mix(r, aug, seed)
        rows.append(r.to(device))

    for kf in payload.get("keyframes") or []:
        if active_kf is None or any(kf is k for k in active_kf):
            _add_row(kf)
    for r in payload.get("refs") or []:
        if "latent" in r and r.get("kind") != "audio":
            if active_refs is None or any(r is x for x in active_refs):
                _add_row(r)
    return torch.cat(rows, dim=0) if rows else None


def apply_cond_video_patch():
    return _install(
        getattr(mm_model, "MiniMaxH3Model", None), "_cond_video_rows",
        _patched_cond_video_rows,
        "per-block video strength unavailable",
        "per-block video strength enabled")
