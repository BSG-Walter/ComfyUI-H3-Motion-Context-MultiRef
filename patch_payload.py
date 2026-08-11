"""Let keyframes and refs coexist, and give per-block audio strength.

`MiniMaxH3.extra_conds` in comfy/model_base.py fills the payload from two
independent `if` blocks. The keyframe block sets `cond_video_latents`, then
the refs block overwrites it. So attaching an audio-only ref alongside
keyframes wipes the keyframe video content. This wrapper re-runs the same
logic and concatenates instead. Graphs using only one mechanism are
unaffected.

Per-block strength: `MiniMaxH3Model._cond_audio_rows` mixes every reference
audio latent with `(1 - audio_cond_noise_aug)` of fresh noise, controlled by
ONE global payload key. Fresh noise is only meaningful above the 50/50
point; below it the injected row is mostly static. Instead, our blocks blend
the pinned clip with the model's OWN evolving audio at those steps: at every
denoising step the block's rows become `s * clip + (1 - s) * generation`,
so strength is a continuous influence fraction: 1.0 pins exactly, 0.5 is
half clip / half model, 0.1 is a hint, 0 is transparent. The target step
indices come from the layout's position times (verified exact to < 1/2
step), and the harness wraps MiniMaxH3Model.forward to stash the evolving
target rows into the payload right before _cond_audio_rows packs them.
"""

import logging

import torch

import comfy.ldm.minimax.model as mm_model
import comfy.model_base as model_base

from .patch_layout import MC_AUDIO_STRENGTH, _recover_foreign

_LOG = logging.getLogger("h3_motion_context")

_ORIG = {}
_warned_no_blend = False

_FOREIGN_ORIG_NAMES = (
    "_orig_extra_conds",
    "_orig_cond_audio_rows",
    "_orig_forward",
    "_original_extra_conds",
    "_original_cond_audio_rows",
    "_original_forward",
    "_stock_extra_conds",
    "_stock_cond_audio_rows",
    "_stock_forward",
)

# payload keys this fork owns; extra_conds leaves them untouched
_BLEND_MAP_KEY = "_h3mc_audio_blend_map"
_BLEND_ROWS_KEY = "_h3mc_audio_blend_rows"


def _install(cls, attr, patched, foreign_names, gone_msg, done_msg):
    """Adopt or install `patched` over `cls.attr`; returns success.

    Idempotent: if our own wrapper (possibly from a previous import of this
    module) already owns the attribute, it is adopted instead of re-wrapped.
    If a wrapper installed by another h3_motion_context copy owns it, the
    stock function it captured is recovered and taken over.
    """
    if cls is None or not hasattr(cls, attr):
        _LOG.warning("h3_motion_context: %s not found, %s", attr, gone_msg)
        return False
    current = getattr(cls, attr)
    if getattr(current, "_h3mc_patcher", False):
        _ORIG[attr] = getattr(current, "_h3mc_orig", current)
        return True
    if current is not patched:
        foreign = _recover_foreign(current, foreign_names)
        if foreign is not None:
            _ORIG[attr] = foreign
            setattr(patched, "_h3mc_patcher", True)
            setattr(patched, "_h3mc_orig", foreign)
            setattr(cls, attr, patched)
            _LOG.warning("h3_motion_context: took over the %s patch installed "
                         "by another h3_motion_context copy", attr)
            return True
    _ORIG[attr] = current
    setattr(patched, "_h3mc_patcher", True)
    setattr(patched, "_h3mc_orig", current)
    setattr(cls, attr, patched)
    _LOG.info("h3_motion_context: %s", done_msg)
    return True


def _patched_extra_conds(self, **kwargs):
    out = _ORIG["extra_conds"](self, **kwargs)
    keyframes = kwargs.get("minimax_keyframes", None)
    refs = kwargs.get("minimax_refs", None)
    if not keyframes or not refs:
        return out  # only one mechanism in play, stock behaviour is correct

    cond = out.get("minimax_payload", None)
    payload = getattr(cond, "cond", None) if cond is not None else None
    if not isinstance(payload, dict):
        _LOG.warning("h3_motion_context: could not reach the H3 payload, "
                     "keyframe latents may have been overwritten by refs")
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
        _patched_extra_conds, _FOREIGN_ORIG_NAMES,
        "keyframes and refs cannot be combined",
        "keyframe/ref coexistence enabled")


def _target_audio_segment(layout):
    """(start_row, steps) of the target audio stream segment in the layout.

    The target stream is the last 'audio'-kind segment; injected blocks are
    'ref_audio'-kind segments, so the kinds never collide.
    """
    tgt = None
    for a, b, kind in layout.segments:
        if kind == "audio":
            tgt = (a, b)
    if tgt is None:
        return None
    return (tgt[0], (tgt[1] - tgt[0]) // 2)


def _audio_blend_map(layout, refs):
    """Map each weak audio ref's steps onto target audio steps.

    Returns {audio_ref_index: [j0, j1, ...]} with one entry per latent step
    of the block (len == ref_audio_t). The packed row space lays the two
    stereo channels SEQUENTIALLY (rows 0..rt-1 are channel 0, rows rt..2rt-1
    channel 1), while the target audio stream is INTERLEAVED (step j lives
    at rows [t0 + 2j, t0 + 2j + 2)). Both families step +1.0 per channel
    step in position_ids, so the pairing of block step k to target step j is
    exact to well under half a step after the layout patch repositions the
    block onto (or before) the clip's own audio timeline; out-of-target
    steps clamp to the nearest edge.
    """
    if layout is None:
        return {}
    tgt = _target_audio_segment(layout)
    if tgt is None:
        return {}
    t0, t_steps = tgt
    t = layout.position_ids[:, 0]
    origin = float(t[t0])
    audio_refs = [i for i, r in enumerate(refs or [])
                  if r.get("audio_latent") is not None]
    it = iter(audio_refs)
    out = {}
    for a, b, kind in layout.segments:
        if kind != "ref_audio":
            continue
        try:
            i = next(it)
        except StopIteration:
            break
        s = refs[i].get(MC_AUDIO_STRENGTH, 1.0)
        if s is None or float(s) >= 1.0:
            continue
        rt = (b - a) // 2
        steps = []
        for k in range(rt):
            tau = float(t[a + k])  # channel 0 row k (channel 1 repeats them)
            j = int(round(tau - origin))
            steps.append(min(max(j, 0), t_steps - 1))
        if steps:
            out[i] = steps
    return out


def _patched_cond_audio_rows(self, payload, device):
    """Stock row packer plus per-block reference strength.

    For every marked block with strength in (0, 1), the injected rows are
    `s * pack(clip) + (1 - s) * target_rows`, where target_rows are the
    model's OWN evolving audio rows at the block's steps, gathered by the
    forward wrapper into the payload right before this runs. Strength is
    thus a continuous influence fraction: 0.9 mostly the clip, 0.5 half,
    0.1 a hint, approaching 0 transparent (the block blends into the
    model's own generation). Strength 1.0 pins exactly; unmarked refs keep
    stock behaviour (global audio_cond_noise_aug mixing).

    refs and cond_audio_latents are both ordered like the refs list (stock
    packs them filtered in order), so the two zip 1:1; any mismatch falls
    back to the stock packer entirely.
    """
    audio_refs = [r for r in payload.get("refs") or []
                  if r.get("audio_latent") is not None]
    latents = payload.get("cond_audio_latents") or []
    if len(latents) != len(audio_refs):
        if latents:
            _LOG.warning("h3_motion_context: cond_audio_latents/refs mismatch "
                         "(%d vs %d), per-block strength disabled this run",
                         len(latents), len(audio_refs))
        return _ORIG["_cond_audio_rows"](self, payload, device)

    stash = payload.get(_BLEND_ROWS_KEY) or {}
    rows = []
    seed = int(payload.get("seed", 0)) + 1
    default_aug = float(payload.get("audio_cond_noise_aug", 1.0))
    for i, (z, ref) in enumerate(zip(latents, audio_refs)):
        s = ref.get(MC_AUDIO_STRENGTH)
        if s is not None:
            s = float(s)
        target = stash.get(i)
        r = mm_model.pack_audio(z.to(torch.float32))
        if target is not None and s is not None and s < 1.0:
            if target.shape[0] == r.shape[0]:
                r = s * r + (1.0 - s) * target.to(r.device)
            else:
                _LOG.warning("h3_motion_context: audio strength target row "
                             "mismatch (%d vs %d), using stock",
                             target.shape[0], r.shape[0])
                r = mm_model.pack_audio(z.to(torch.float32))
        elif s is not None and s >= 1.0:
            pass  # exact pin, stock rows
        elif default_aug < 1.0:
            gen = torch.Generator("cpu").manual_seed(seed)
            noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)
            r = default_aug * r + (1.0 - default_aug) * noise.to(r.device)
        rows.append(r.to(device))
    return torch.cat(rows, dim=0) if rows else None


def apply_cond_audio_patch():
    return _install(
        getattr(mm_model, "MiniMaxH3Model", None), "_cond_audio_rows",
        _patched_cond_audio_rows, _FOREIGN_ORIG_NAMES,
        "per-block audio strength unavailable",
        "per-block audio strength enabled")


def _patched_forward(self, x, timestep, context, transformer_options={},
                     minimax_payload=None, **kwargs):
    """Stash the model's own evolving audio rows for low-strength blocks.

    Pinned audio ref rows are never denoised and are injected into the audio
    stream at every step. For blocks whose strength is below 1.0 the injected
    content must be a strength-weighted blend of the clip and what the model
    itself is currently generating at those steps; that generation lives in
    x[1] (the evolving target audio latent), packed exactly like the rows the
    stock packer will emit. This wrapper computes the     per-block target step
    map once per sampling run, gathers the evolving rows every forward call,
    and stashes them in the payload for _patched_cond_audio_rows.
    """
    global _warned_no_blend
    payload = minimax_payload or {}
    blend = None
    if payload.get(_BLEND_MAP_KEY) is None:
        if isinstance(x, (list, tuple)) and len(x) > 1 and x[1] is not None \
                and getattr(x[1], "ndim", 0) == 4:
            blend = _audio_blend_map(payload.get("layout"),
                                    payload.get("refs") or [])
            if blend:
                _LOG.info("h3_motion_context: audio strength is now a "
                          "continuous influence blend (%d blocks)",
                          len(blend))
        elif not _warned_no_blend:
            _warned_no_blend = True
            _LOG.warning("h3_motion_context: forward got no 4-D audio "
                         "latent, continuous blending unavailable this run")
        payload[_BLEND_MAP_KEY] = blend
    blend = payload.get(_BLEND_MAP_KEY)
    if blend and isinstance(x, (list, tuple)) and len(x) > 1 \
            and x[1] is not None and getattr(x[1], "ndim", 0) == 4:
        audio_rows = mm_model.pack_audio(x[1].to(torch.float32))
        tgt = _target_audio_segment(payload.get("layout"))
        t0 = tgt[0] if tgt is not None else None
        stash = {}
        if t0 is not None:
            for i, steps in blend.items():
                # blocks lay their two channels sequentially (row m of the
                # packed block = channel m//rt, step m%rt); the target
                # interleaves (step j at rows 2j, 2j+1 of the packed
                # x[1], whose rows are the stream's target rows starting
                # at t0). Keep the stash in the block's own row order so
                # the element-wise mix in _patched_cond_audio_rows pairs
                # channels and steps exactly.
                rt = len(steps)
                idx = [2 * steps[m % rt] + (m // rt) for m in range(rt * 2)]
                pair = torch.tensor(idx, dtype=torch.long)
                sel = audio_rows.index_select(0, pair.to(audio_rows.device))
                stash[i] = sel.reshape(rt * 2, audio_rows.shape[1])
        payload[_BLEND_ROWS_KEY] = stash
    return _ORIG["forward"](self, x, timestep, context,
                            transformer_options=transformer_options,
                            minimax_payload=minimax_payload, **kwargs)


def apply_forward_patch():
    return _install(
        getattr(mm_model, "MiniMaxH3Model", None), "forward",
        _patched_forward, _FOREIGN_ORIG_NAMES,
        "continuous audio strength blending unavailable",
        "continuous audio strength blending enabled")
