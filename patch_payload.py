"""Let keyframes and refs coexist, and give per-block strength.

`MiniMaxH3.extra_conds` in comfy/model_base.py fills the payload from two
independent `if` blocks. The keyframe block sets `cond_video_latents`, then
the refs block overwrites it. So attaching an audio-only ref alongside
keyframes wipes the keyframe video content. This wrapper re-runs the same
logic and concatenates instead. Graphs using only one mechanism are
unaffected.

Per-block strength (video AND audio) is a pin-then-flip schedule. The
model is told how noisy each condition is by the timestep it assigns to
the condition tokens: stock gives every video condition block
`max(t_v, visual_cond_noise_aug)` and every audio ref block
`max(t_a, audio_cond_noise_aug)`. Mixing per-block rows at a custom
strength while leaving the claim at the stock value is a lie: the model
treats the half-noisy rows as clean content and bakes the noise into the
output (video grain below ~0.75, audio noise on low-denoise runs). So a
marked weak block is pinned EXACT while its timeline's progress stays
below its strength (rows unmixed, claim forced to 0.999 - the canonical
image-to-video pair), then its tokens are dropped from the layout for the
rest of the run (`t_v >= s` for video, `t_a >= s` for audio), so the
model's own stream covers the region with no reference at all. Strength is
the fraction of the run the block stays pinned: 1.0 exact, 0.5 pinned
half then free, 0.0 a pure prompt block. Nothing noisy is ever shown to
the model. Unmarked blocks keep stock behaviour in both packers.
"""

import logging

import torch

import comfy.ldm.common_dit
import comfy.ldm.minimax.model as mm_model
import comfy.model_base as model_base

from .patch_layout import (
    MC_AUDIO_STRENGTH,
    MC_VIDEO_STRENGTH,
    _FOREIGN_ORIG_NAMES,
    _recover_foreign,
)

_LOG = logging.getLogger("h3_motion_context")

_ORIG = {}

# module-global names a foreign wrapper of the payload attributes may have
# captured the stock functions under, on top of the layout ones
_FOREIGN_ORIG_NAMES = _FOREIGN_ORIG_NAMES + (
    "_orig_extra_conds",
    "_orig_cond_audio_rows",
    "_orig_cond_video_rows",
    "_original_extra_conds",
    "_original_cond_audio_rows",
    "_original_cond_video_rows",
    "_stock_extra_conds",
    "_stock_cond_audio_rows",
    "_stock_cond_video_rows",
)


def _mix(r, aug, seed):
    """Stock global-aug row mixing for unmarked blocks."""
    if aug < 1.0:
        gen = torch.Generator("cpu").manual_seed(seed)
        noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)
        r = aug * r + (1.0 - aug) * noise.to(r.device)
    return r


def _install(cls, attr, patched, foreign_names, gone_msg, done_msg):
    """Adopt or install `patched` over `cls.attr`; returns success.

    Idempotent: if our own wrapper (possibly from a previous import of this
    module) already owns the attribute, it is adopted instead of re-wrapped.
    If a wrapper installed by another h3_motion_context copy owns it, the
    patch is refused: the docs say to delete the other copy.
    """
    if cls is None or not hasattr(cls, attr):
        _LOG.warning("h3_motion_context: %s not found, %s", attr, gone_msg)
        return False
    current = getattr(cls, attr)
    if getattr(current, "_h3mc_patcher", False):
        _ORIG[attr] = getattr(current, "_h3mc_orig", current)
        return True
    if current is not patched and _recover_foreign(
            current, foreign_names) is not None:
        _LOG.warning("h3_motion_context: another h3_motion_context copy has "
                     "already patched %s; DELETE every other copy and "
                     "restart ComfyUI.", attr)
        return False
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
        return out  # only one mechanism in play, stock behaviour is correct

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
        _patched_extra_conds, _FOREIGN_ORIG_NAMES,
        "keyframes and refs cannot be combined",
        "keyframe/ref coexistence enabled")


def _patched_cond_audio_rows(self, payload, device):
    """Stock audio row packer plus per-ref pin/flip semantics.

    The latents are read straight off the ref dicts (never from the
    cond_audio_latents list, which the wrapper leaves untouched), and any
    ref dropped by the forward wrapper is skipped. A marked ref is pinned
    EXACT (clean rows, no mixing) - the wrapper claims it clean (0.999)
    and drops it from the layout once the audio timeline crosses its
    strength. Unmarked refs keep stock behaviour (global
    audio_cond_noise_aug mixing).
    """
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
        _patched_cond_audio_rows, _FOREIGN_ORIG_NAMES,
        "per-block audio strength unavailable",
        "per-block audio strength enabled")


def _patched_forward(self, x, timestep, context, transformer_options={},
                     minimax_payload=None, **kwargs):
    """Pin-then-flip per-block strength (video keyframes and audio refs).

    The model learns how noisy a condition is from the timestep assigned to
    the condition tokens, so custom-noise rows under a stock claim are a
    lie and the noise bakes into the output. Instead a marked weak block is
    pinned EXACT while its timeline's progress stays below the block's
    strength: rows unmixed, claim forced to 0.999 - the canonical
    image-to-video pair. Once progress crosses the strength, the block is
    dropped from the layout for the rest of the run (its tokens removed, so
    the model's own stream covers the region with no reference at all).
    Video progress is t_v = 1 - sigma; audio progress is t_a on the audio's
    own shifted schedule. Strength is the fraction of the run the block
    stays pinned: 1.0 exact, 0.5 pinned half then free, 0.0 a pure prompt
    block. Nothing noisy is ever shown to the model.

    The payload lists (keyframes/refs/cond_*_latents) are NEVER mutated:
    the active set is stored as `_h3mc_active_keyframes` / `_h3mc_active_refs`
    (the same dict objects), the layout is rebuilt only when the set changes,
    and the patched packers read the latents straight off the dicts and skip
    inactive blocks. Consistency is structural, so there is no list
    bookkeeping to desync. Unmarked runs pass straight through untouched.
    """
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
            payload["layout"] = mm_model.PackedLayout(
                context.shape[1], video_x.shape[2], video_x.shape[3],
                video_x.shape[4], x[1].shape[-1],
                keyframes=keep_kf, refs=keep_refs,
                frame_count=payload.get("frame_count"))
            payload["_h3mc_active_keyframes"] = keep_kf
            payload["_h3mc_active_refs"] = keep_refs
    out = _ORIG["forward"](self, x, timestep, context,
                           transformer_options, minimax_payload, **kwargs)
    _clamp_hard(out, x, timestep, payload)
    return out


def _clamp_hard(out, x, timestep, payload):
    """Hard-inject pinned video steps into the sampler output.

    The forward returns [-video_out, -audio_out] and the flow sampler
    forms the denoised estimate as x - out[0] * sigma (CONST schedule), so
    setting out[0] at a step to (video_x - lat) / sigma makes the estimate
    exactly the pinned latent at that step, every step of the chain. Both
    the cond and uncond CFG passes see the same clamped values, so the
    combination keeps the pins exact and the rest of the video is sampled
    normally. Content from a pinned step can never regenerate, no matter
    what the model's attention would rather do.
    """
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
        _patched_forward, _FOREIGN_ORIG_NAMES,
        "per-keyframe video strength unavailable",
        "per-keyframe video strength enabled")


def _patched_cond_video_rows(self, payload, device):
    """Stock video row packer plus per-keyframe pin/flip semantics.

    Same idea as `_patched_cond_audio_rows`: rows are read straight off the
    keyframe/ref dicts (never from cond_video_latents, which the wrapper
    leaves untouched), and any block dropped by the forward wrapper is
    skipped. A marked weak block is pinned EXACT (clean rows, no mixing);
    the wrapper claims it clean (0.999) and drops it from the layout once
    the video timeline crosses its strength. Unmarked blocks keep the stock
    global-aug mixing path. Unmarked runs take the stock packer wholesale.
    """
    active_kf = payload.get("_h3mc_active_keyframes")
    active_refs = payload.get("_h3mc_active_refs")
    if active_kf is None and active_refs is None:
        return _ORIG["_cond_video_rows"](self, payload, device)
    rows = []
    aug = float(payload.get("visual_cond_noise_aug",
                            mm_model.VISUAL_COND_TIMESTEP))
    seed = int(payload.get("seed", 0))
    for kf in payload.get("keyframes") or []:
        if active_kf is not None and not any(kf is k for k in active_kf):
            continue
        r = mm_model.patchify_video(kf["latent"].to(torch.float32),
                                    self.patch_size)
        if kf.get(MC_VIDEO_STRENGTH) is None:
            r = _mix(r, aug, seed)
        rows.append(r.to(device))
    for r in payload.get("refs") or []:
        if "latent" not in r or r.get("kind") == "audio":
            continue  # kind "audio" gets only a ref_audio segment in the layout
        if active_refs is not None and not any(r is x for x in active_refs):
            continue
        r2 = mm_model.patchify_video(r["latent"].to(torch.float32),
                                     self.patch_size)
        if r.get(MC_VIDEO_STRENGTH) is None:
            r2 = _mix(r2, aug, seed)
        rows.append(r2.to(device))
    return torch.cat(rows, dim=0) if rows else None


def apply_cond_video_patch():
    return _install(
        getattr(mm_model, "MiniMaxH3Model", None), "_cond_video_rows",
        _patched_cond_video_rows, _FOREIGN_ORIG_NAMES,
        "per-block video strength unavailable",
        "per-block video strength enabled")
