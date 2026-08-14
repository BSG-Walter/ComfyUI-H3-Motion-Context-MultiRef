# MODIFIED FORK NOTICE: modified 2026-08-09 to allow ordinary MiniMax H3 Ref2VA refs
# to coexist with H3 Motion Context timeline-audio refs. Original project by NikoDemon80.
# See MODIFICATIONS.md. Distributed under the upstream GPL-3.0 license.

"""Lift MiniMax H3's first/last-only keyframe anchor restriction.

General position formula:
    cond_t = text_len + FRAME_RESCALE * pixel_index
"""

import logging
import comfy.ldm.minimax.model as mm

MC_KEY = "motion_context_index"
MC_AUDIO_KEY = "motion_context_audio_end_frame"
MC_AUDIO_STRENGTH = "motion_context_audio_strength"
MC_VIDEO_STRENGTH = "motion_context_video_strength"
_LOG = logging.getLogger("h3_motion_context")

_orig_init = None
_applied = False


def _ref_cursor_advance(refs):
    """How far ref blocks push the target origin past text_len."""
    if not refs:
        return 0.0
    cursor = 0.0
    for blk in refs:
        kind = blk.get("kind")
        if kind == "image":
            cursor += 1.0
        elif kind == "audio":
            cursor += float(blk.get("ref_audio_t", 0))
        elif kind in ("video", "video_audio"):
            rt = float(blk.get("ref_audio_t", 0))
            vt = int(blk.get("latent_t", 0))
            cursor += max(rt, sum(mm._video_t_spans(vt)))
    return cursor


def _cond_t(text_len, latent_t, frame_count, p):
    """Time coordinate for a keyframe anchored at pixel frame p."""
    return float(text_len) + mm.FRAME_RESCALE * float(p)


def _fixup(layout, text_len, latent_t, frame_count, keyframes, refs=None):
    """Rewrite cond-row time coordinates to the general position formula."""
    offset = _ref_cursor_advance(refs)
    if offset and any(kf.get(MC_KEY) is None for kf in keyframes):
        raise RuntimeError(
            "h3_motion_context: stock and motion-context keyframes mixed in "
            "one graph alongside a ref; their coordinates would disagree. "
            "Give every keyframe a %s entry or remove the refs." % MC_KEY)
    cond_spans = [(a, b) for a, b, kind in layout.segments if kind == "cond"]
    if len(cond_spans) != len(keyframes):
        raise RuntimeError(
            "h3_motion_context: expected %d cond segments, layout has %d. "
            "Refusing to rewrite positions."
            % (len(keyframes), len(cond_spans)))
    for (a, b), kf in zip(cond_spans, keyframes):
        p = kf.get(MC_KEY)
        if p is None:
            continue
        layout.position_ids[a:b, 0] = _cond_t(text_len, latent_t, frame_count, p) + offset


def _fixup_audio(layout, text_len, refs):
    """Move every marked audio ref onto the target timeline."""
    marked = [(i, r) for i, r in enumerate(refs)
              if r.get(MC_AUDIO_KEY) is not None]
    if not marked:
        return
    for i, blk in marked:
        if blk.get("kind") != "audio":
            raise RuntimeError(
                "h3_motion_context: %s set on a %r ref; only audio refs can be "
                "moved onto the timeline." % (MC_AUDIO_KEY, blk.get("kind")))

    t = layout.position_ids[:, 0]
    snapshot = t.clone()
    eps = 1e-4
    cond_rows = set()
    for a, b, kind in layout.segments:
        if kind == "cond":
            cond_rows.update(range(a, b))

    target_origin = float(text_len) + _ref_cursor_advance(refs)

    for i, blk in marked:
        rt = int(blk.get("ref_audio_t", 0))
        if rt <= 0:
            continue

        prefix = _ref_cursor_advance(refs[:i])
        slot_start = float(text_len) + prefix
        slot_end = slot_start + float(rt)

        sel = (snapshot >= slot_start - eps) & (snapshot < slot_end - eps)
        for r in cond_rows:
            sel[r] = False

        count = int(sel.sum())
        if count < rt or count > 8 * rt:
            raise RuntimeError(
                "h3_motion_context: found %d rows in marked audio ref slot "
                "[%.4f, %.4f) for %d latent steps, expected between %d and %d. "
                "Upstream layout change or overlapping coordinates; refusing "
                "to move audio rows."
                % (count, slot_start, slot_end, rt, rt, 8 * rt))

        end_frame = float(blk[MC_AUDIO_KEY])
        desired_start = target_origin + mm.FRAME_RESCALE * end_frame - float(rt)
        shift = desired_start - slot_start
        t[sel] = snapshot[sel] + shift


def _init_layout(layout, text_len, latent_t, lh, lw, audio_t, keyframes=None, refs=None, frame_count=None):
    kw = {}
    if keyframes is not None:
        kw["keyframes"] = keyframes
    if refs is not None:
        kw["refs"] = refs
    if frame_count is not None and "frame_count" in getattr(_orig_init, "__code__", object()).co_varnames:
        kw["frame_count"] = frame_count
    _orig_init(layout, text_len, latent_t, lh, lw, audio_t, **kw)


def _patched_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
                  keyframes=None, refs=None, frame_count=None, **kwargs):
    _init_layout(self, text_len, latent_t, latent_h, latent_w, audio_t,
                 keyframes=keyframes, refs=refs, frame_count=frame_count)
    if keyframes and any(kf.get(MC_KEY) is not None for kf in keyframes):
        _fixup(self, text_len, latent_t, frame_count, keyframes, refs)
    if refs and any(r.get(MC_AUDIO_KEY) is not None for r in refs):
        _fixup_audio(self, text_len, refs)


def apply_patch():
    global _orig_init, _applied
    if _applied:
        return True
    if not hasattr(mm, "PackedLayout") or not hasattr(mm, "FRAME_RESCALE"):
        _LOG.warning("h3_motion_context: MiniMax H3 model module missing expected "
                     "attributes, patch not applied")
        return False
    current = mm.PackedLayout.__init__
    if current is not _patched_init and getattr(current, "_h3mc_layout_patcher", False):
        _orig_init = current._h3mc_orig_init
        _applied = True
        return True
    _orig_init = current
    _patched_init._h3mc_orig_init = _orig_init
    _patched_init._h3mc_layout_patcher = True
    mm.PackedLayout.__init__ = _patched_init
    _applied = True
    _LOG.info("h3_motion_context: interior keyframe anchors enabled")
    return True
