# MODIFIED FORK NOTICE: modified 2026-08-09 to allow ordinary MiniMax H3 Ref2VA refs
# to coexist with H3 Motion Context timeline-audio refs. Original project by NikoDemon80.
# See MODIFICATIONS.md. Distributed under the upstream GPL-3.0 license.

"""Lift MiniMax H3's first/last-only keyframe anchor restriction.

Stock ComfyUI builds keyframe conditioning rows at one of two time
coordinates and rejects everything else:

    if pixel_index == 0:
        cond_t = float(text_len)
    elif frame_count is not None and pixel_index == frame_count - 1:
        cond_t = float(text_len) + sum(_video_t_spans(latent_t)) - FRAME_RESCALE
    else:
        raise ValueError("only first/last keyframe anchors are supported")

Both branches are the same expression. Each video token spans
FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] and covers FRAME_PER_TOKEN[k % 5]
pixel frames, so the cumulative time at pixel frame p is exactly
FRAME_RESCALE * p, for every p. Substituting p = frame_count - 1
reproduces the second branch identically:

    text_len + FRAME_RESCALE * (frame_count - 1)
      == text_len + FRAME_RESCALE * frame_count - FRAME_RESCALE
      == text_len + sum(_video_t_spans(latent_t)) - FRAME_RESCALE

So the general position is:

    cond_t = text_len + FRAME_RESCALE * pixel_index

We do NOT rewrite the source of PackedLayout.__init__. Instead every
keyframe is handed to stock code with resolved_frame_index = 0, which is
always legal, and the real index rides along under MC_KEY. After the
stock constructor returns we rewrite the time column of each cond
segment's rows in position_ids. RoPE is built at forward time from
position_ids, so this lands before anything reads it.

That keeps the patch surface to one attribute we can verify rather than a
copy of a 90-line constructor that would rot on the next ComfyUI change.
"""

import logging
import os

import torch

import comfy.ldm.minimax.model as mm

MC_KEY = "motion_context_index"
MC_AUDIO_KEY = "motion_context_audio_end_frame"
MC_AUDIO_STRENGTH = "motion_context_audio_strength"
MC_VIDEO_STRENGTH = "motion_context_video_strength"
_LOG = logging.getLogger("h3_motion_context")

# Module-global names older copies of this code (or of the upstream package)
# capture the stock init under when they install their own wrapper. A wrapper
# we do not recognize as ours may still expose the stock constructor under
# one of these names in its module globals.
_FOREIGN_ORIG_NAMES = (
    "_orig_init",
    "_original_init",
    "_stock_init",
    "_unpatched_init",
    "_orig_initializer",
    "_base_init",
)

_orig_init = None
_applied = False


def _recover_foreign(attr, names):
    """Recover a stock function a foreign wrapper captured under any of the
    candidate module-global names."""
    globs = getattr(attr, "__globals__", None)
    if isinstance(globs, dict):
        get = globs.get
        for name in names:
            cand = get(name)
            if callable(cand) and cand is not attr:
                return cand
    for name in names:
        cand = getattr(attr, name, None)
        if callable(cand) and cand is not attr:
            return cand
    return None


def _find_dup_installs():
    """Folder names of other H3-Motion-Context copies in custom_nodes."""
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.dirname(here)
    if not os.path.isdir(base):
        return []
    try:
        names = os.listdir(base)
    except OSError:
        return []
    own = os.path.basename(here).lower()
    found = []
    for name in names:
        if "h3-motion-context" in name.lower() and name.lower() != own:
            found.append(name)
    return sorted(found)


def _ref_cursor_advance(refs):
    """How far ref blocks push the target origin past text_len.

    Refs are laid out sequentially from a cursor that starts at text_len,
    and the target audio and video rows use the cursor's final value as
    their origin. Keyframe coordinates are computed from text_len directly,
    so without this term adding any ref would slide the anchors backwards
    relative to the clip they are anchoring.
    """
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
    """Time coordinate for a keyframe anchored at pixel frame p.

    The endpoints reuse stock's exact expressions rather than the general
    formula. They are mathematically identical, but stock accumulates
    latent_t float additions where the general form does one multiply, and
    those differ in the last bits (about 7e-15). Matching stock bit for bit
    means an existing first/last graph builds byte-identical positions
    after this patch is applied, and lets the self-test stay strict.
    """
    if p == 0:
        return float(text_len)
    if frame_count is not None and p == frame_count - 1:
        return float(text_len) + sum(mm._video_t_spans(latent_t)) - mm.FRAME_RESCALE
    return float(text_len) + mm.FRAME_RESCALE * float(p)


def _fixup(layout, text_len, latent_t, frame_count, keyframes, refs=None):
    """Rewrite cond-row time coordinates to the general position formula."""
    offset = _ref_cursor_advance(refs)
    if offset and any(kf.get(MC_KEY) is None for kf in keyframes):
        # keyframes without MC_KEY are left exactly as stock built them,
        # which means they do NOT get the ref cursor compensation. Mixing
        # them with MC keyframes under a ref would slide the stock anchors
        # relative to ours and to the target. Nothing produces this today;
        # refuse loudly in case something ever does.
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
    """Move every marked audio ref onto the target timeline.

    Each ref marked with MC_AUDIO_KEY is shifted so its block ends at
    target_origin + FRAME_RESCALE * end_frame, which places it anywhere on
    (or before) the clip's own audio timeline -- beginning, middle or end,
    exactly how keyframe coordinates anchor the video. Ordinary Ref2VA refs
    keep their stock slots.

    Marked refs may appear anywhere among the refs. Every slot selection is
    taken from a pre-shift snapshot of the time coordinates, so moving one
    block can never corrupt the slot detection of another, even when two
    blocks end up overlapping on the timeline.
    """
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

def _assert_moved(td, te, slots, cond_rows, want_shift=None):
    """Self-test helper: exactly the rows whose pre-shift coordinate falls in
    one of `slots` moved, each by the same shift."""
    moved = set(i for i in range(len(td)) if float(td[i]) != float(te[i]))
    expect = set()
    for start, end in slots:
        expect.update(i for i in range(len(td))
                      if start - 1e-4 <= float(td[i]) < end - 1e-4
                      and i not in cond_rows)
    if moved != expect:
        raise RuntimeError(
            "audio move touched the wrong rows: %d moved, %d expected, "
            "e.g. %s" % (len(moved), len(expect),
                         sorted(moved ^ expect)[:8]))
    if not moved:
        raise RuntimeError("audio move moved no rows")
    if want_shift is not None:
        deltas = [float(te[i]) - float(td[i]) for i in sorted(moved)]
        if any(abs(dd - want_shift) > 1e-5 for dd in deltas):
            raise RuntimeError(
                "audio rows shifted non-uniformly or by the wrong amount: "
                "%s vs %.6f" % (deltas[:4], want_shift))

def _patched_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
                  keyframes=None, refs=None, frame_count=None):
    _orig_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
               keyframes=keyframes, refs=refs, frame_count=frame_count)
    has_mc_kf = bool(keyframes) and any(
        kf.get(MC_KEY) is not None for kf in keyframes)
    has_mc_audio = bool(refs) and any(
        r.get(MC_AUDIO_KEY) is not None for r in refs)
    if has_mc_kf:
        _fixup(self, text_len, latent_t, frame_count, keyframes, refs)
    if has_mc_audio:
        _fixup_audio(self, text_len, refs)
    # neither marked: stock graph, leave it exactly as built


def _self_test():
    """Prove the rewrite reproduces stock positions before committing.

    Builds the two anchors stock code already supports, once the stock way
    and once through our mechanism, and requires the position tensors to
    match exactly. If ComfyUI changes the position maths underneath us this
    fails and the patch is not applied.
    """
    text_len, latent_t, lh, lw, audio_t = 7, 7, 22, 38, 16
    frame_count = sum(mm.FRAME_PER_TOKEN[k % 5] for k in range(latent_t))

    stock_kf = [{"resolved_frame_index": 0},
                {"resolved_frame_index": frame_count - 1}]
    ours_kf = [{"resolved_frame_index": 0, MC_KEY: 0},
               {"resolved_frame_index": 0, MC_KEY: frame_count - 1}]

    a = mm.PackedLayout.__new__(mm.PackedLayout)
    _orig_init(a, text_len, latent_t, lh, lw, audio_t,
               keyframes=stock_kf, frame_count=frame_count)

    b = mm.PackedLayout.__new__(mm.PackedLayout)
    _orig_init(b, text_len, latent_t, lh, lw, audio_t,
               keyframes=ours_kf, frame_count=frame_count)
    _fixup(b, text_len, latent_t, frame_count, ours_kf)

    if a.position_ids.shape != b.position_ids.shape:
        raise RuntimeError("position_ids shape mismatch in self-test")
    if not torch.equal(a.position_ids, b.position_ids):
        bad = (a.position_ids != b.position_ids).any(dim=1).nonzero().flatten()
        raise RuntimeError("position mismatch at rows %s" % bad[:8].tolist())

    # a consecutive run must land on strictly increasing coordinates inside
    # the span the two endpoints define
    run = [{"resolved_frame_index": 0, MC_KEY: i} for i in range(4)]
    c = mm.PackedLayout.__new__(mm.PackedLayout)
    _orig_init(c, text_len, latent_t, lh, lw, audio_t,
               keyframes=run, frame_count=frame_count)
    _fixup(c, text_len, latent_t, frame_count, run)
    ts = [float(c.position_ids[s, 0]) for s, _, k in c.segments if k == "cond"]
    if len(ts) != len(run):
        raise RuntimeError("expected %d cond segments, got %d" % (len(run), len(ts)))
    if any(ts[i] >= ts[i + 1] for i in range(len(ts) - 1)):
        raise RuntimeError("consecutive anchors not strictly increasing: %s" % ts)
    t_last = float(text_len) + mm.FRAME_RESCALE * (frame_count - 1)
    if not (ts[0] == float(text_len) and ts[-1] < t_last):
        raise RuntimeError("run %s escapes the [%.4f, %.4f] span"
                           % (ts, float(text_len), t_last))

    # adding a ref must not move the anchors relative to the target. Stock
    # cond rows cannot be the reference here: stock computes them from
    # text_len and never compensates for refs, which is the very bug
    # _ref_cursor_advance exists to fix. The ground truth is the target
    # rows themselves. Ref rows are laid out BEFORE the target, so the
    # largest time coordinate in position_ids belongs to the end of the
    # target in both layouts, and the anchor-to-end gap must be identical
    # with and without the ref. This exercises _ref_cursor_advance against
    # stock's real cursor arithmetic, so if upstream changes how refs
    # advance the cursor, this fails and the patch is not applied.
    ref = [{"kind": "audio", "ref_audio_t": 8}]
    d = mm.PackedLayout.__new__(mm.PackedLayout)
    _orig_init(d, text_len, latent_t, lh, lw, audio_t,
               keyframes=run, refs=ref, frame_count=frame_count)
    _fixup(d, text_len, latent_t, frame_count, run, refs=ref)
    ts_ref = [float(d.position_ids[s, 0]) for s, _, k in d.segments if k == "cond"]
    if len(ts_ref) != len(ts):
        raise RuntimeError("cond segment count changed when a ref was added")
    # a semantic failure here is a shift of whole rows (the 8.0 of the ref,
    # or FRAME_RESCALE multiples), while legitimate noise is float
    # accumulation from a different origin, orders of magnitude below 1e-3
    # even at float32. Strict equality stays reserved for the endpoint test.
    tol = 1e-3
    gap = float(c.position_ids[:, 0].max()) - ts[0]
    gap_ref = float(d.position_ids[:, 0].max()) - ts_ref[0]
    if abs(gap - gap_ref) > tol:
        raise RuntimeError(
            "ref compensation off by %.6f: anchor-to-target gap %.6f without "
            "ref, %.6f with. _ref_cursor_advance no longer matches the "
            "layout's cursor arithmetic." % (gap_ref - gap, gap, gap_ref))
    shifts = [b - a for a, b in zip(ts, ts_ref)]
    if any(abs(s - shifts[0]) > tol for s in shifts):
        raise RuntimeError("ref shifted anchors unevenly: %s" % shifts)

    # audio timeline placement: rebuild layout d with the ref marked and
    # require that exactly the rows in the ref's coordinate slot moved,
    # all by one uniform shift, with every other row bit-identical.
    end_frame = 4
    rt = 8
    ref_mc = [{"kind": "audio", "ref_audio_t": rt, MC_AUDIO_KEY: end_frame}]
    e = mm.PackedLayout.__new__(mm.PackedLayout)
    _orig_init(e, text_len, latent_t, lh, lw, audio_t,
               keyframes=run, refs=ref_mc, frame_count=frame_count)
    _fixup(e, text_len, latent_t, frame_count, run, refs=ref_mc)
    _fixup_audio(e, text_len, ref_mc)
    if e.position_ids.shape != d.position_ids.shape:
        raise RuntimeError("audio move changed the layout shape")
    if not torch.equal(d.position_ids[:, 1:], e.position_ids[:, 1:]):
        raise RuntimeError("audio move touched a non-time coordinate column")
    td, te = d.position_ids[:, 0], e.position_ids[:, 0]
    cond_rows = set()
    for a, b, kind in d.segments:
        if kind == "cond":
            cond_rows.update(range(a, b))
    # advance == rt cancels here, so the shift is just the frame offset
    _assert_moved(td, te, [(text_len, text_len + rt)], cond_rows,
                  want_shift=mm.FRAME_RESCALE * end_frame)

    # Two ordinary image refs followed by the marked MC timeline-audio ref.
    img1 = {"kind": "image", "latent_h": lh, "latent_w": lw}
    img2 = {"kind": "image", "latent_h": lh, "latent_w": lw}
    audio_plain = {"kind": "audio", "ref_audio_t": rt}
    audio_marked = {"kind": "audio", "ref_audio_t": rt,
                    MC_AUDIO_KEY: end_frame}
    refs_plain = [img1, img2, audio_plain]
    refs_marked = [img1, img2, audio_marked]

    f = mm.PackedLayout.__new__(mm.PackedLayout)
    _orig_init(f, text_len, latent_t, lh, lw, audio_t,
               keyframes=run, refs=refs_plain, frame_count=frame_count)
    _fixup(f, text_len, latent_t, frame_count, run, refs=refs_plain)

    g = mm.PackedLayout.__new__(mm.PackedLayout)
    _orig_init(g, text_len, latent_t, lh, lw, audio_t,
               keyframes=run, refs=refs_marked, frame_count=frame_count)
    _fixup(g, text_len, latent_t, frame_count, run, refs=refs_marked)
    _fixup_audio(g, text_len, refs_marked)

    if g.position_ids.shape != f.position_ids.shape:
        raise RuntimeError("multi-ref audio move changed the layout shape")
    if not torch.equal(f.position_ids[:, 1:], g.position_ids[:, 1:]):
        raise RuntimeError("multi-ref audio move touched a non-time coordinate")

    tf, tg = f.position_ids[:, 0], g.position_ids[:, 0]
    prefix = _ref_cursor_advance(refs_plain[:2])
    slot_start = float(text_len) + prefix

    cond_rows = set()
    for a, b, kind in f.segments:
        if kind == "cond":
            cond_rows.update(range(a, b))

    target_origin = float(text_len) + _ref_cursor_advance(refs_marked)
    want_multi_shift = (target_origin + mm.FRAME_RESCALE * end_frame
                        - float(rt) - slot_start)
    _assert_moved(tf, tg, [(slot_start, slot_start + float(rt))], cond_rows,
                  want_shift=want_multi_shift)

    # Several marked audio refs at different positions on the same timeline
    # (beginning / middle injection). Each block must end exactly at
    # target_origin + FRAME_RESCALE * end_frame, its rows land on the
    # expected step coordinates, and every other row must stay bit-identical
    # to a graph whose refs were never marked.
    ends = [0, 4, 7]
    rt2 = 6
    refs_two = [{"kind": "audio", "ref_audio_t": rt2, MC_AUDIO_KEY: e}
                for e in ends]
    h = mm.PackedLayout.__new__(mm.PackedLayout)
    _orig_init(h, text_len, latent_t, lh, lw, audio_t,
               keyframes=run, refs=refs_two, frame_count=frame_count)
    _fixup(h, text_len, latent_t, frame_count, run, refs=refs_two)
    _fixup_audio(h, text_len, refs_two)

    i_plain = mm.PackedLayout.__new__(mm.PackedLayout)
    _orig_init(i_plain, text_len, latent_t, lh, lw, audio_t,
               keyframes=run, refs=refs_two, frame_count=frame_count)
    _fixup(i_plain, text_len, latent_t, frame_count, run, refs=refs_two)

    origin2 = float(text_len) + _ref_cursor_advance(refs_two)
    th = h.position_ids[:, 0]
    ref_audio_segs = [(a, b) for a, b, kind in h.segments
                      if kind == "ref_audio"]
    if len(ref_audio_segs) != len(refs_two):
        raise RuntimeError(
            "multi-mark audio: expected %d ref_audio segments, layout has %d"
            % (len(refs_two), len(ref_audio_segs)))
    for (e, r), (a, b) in zip(zip(ends, refs_two), ref_audio_segs):
        # each block's rows are a contiguous packed range; the video/audio
        # target rows legitimately share time coordinates with injected
        # blocks, so match only inside this block's own row range
        rt_i = int(r["ref_audio_t"])
        start = origin2 + mm.FRAME_RESCALE * float(e) - float(rt_i)
        steps = [start + k for k in range(rt_i)]
        for value in steps:
            # float64 additions differing in the last ulp are fine: match
            # within 1e-7. Each step contributes exactly two rows, one per
            # stereo channel, at the same coordinate.
            hit = sum(1 for i in range(a, b)
                      if abs(float(th[i]) - value) < 1e-7)
            if hit != 2:
                raise RuntimeError(
                    "multi-mark audio: coordinate %.9f of block ending at "
                    "frame %d found on %d of its own rows, expected 2 "
                    "(stereo)" % (value, e, hit))
    if not torch.equal(h.position_ids[:, 1:], i_plain.position_ids[:, 1:]):
        raise RuntimeError("multi-mark audio move touched a non-time coordinate")
    cond_rows = set()
    for a, b, kind in h.segments:
        if kind == "cond":
            cond_rows.update(range(a, b))
    # each marked block's rows moved off its stock slot; nothing else did.
    # The shifts are per-block (non-uniform), so no want_shift here.
    _assert_moved(i_plain.position_ids[:, 0], th,
                  [(float(text_len) + j * rt2, float(text_len) + (j + 1) * rt2)
                   for j in range(len(refs_two))], cond_rows)


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
        # OUR wrapper is already installed (the same package imported twice,
        # or a node reload re-ran us). Adopt it instead of wrapping the
        # wrapper: nesting fixups would move rows twice and the self-test
        # would see a corrupted layout.
        _orig_init = current._h3mc_orig_init
        _applied = True
        _LOG.info("h3_motion_context: adopting the already-installed layout patch")
        return True
    if current is not _patched_init and _recover_foreign(
            current, _FOREIGN_ORIG_NAMES) is not None:
        # Another copy of this code (or the upstream package) wrapped the
        # stock init first. Wrapping the wrapper would run the fixups twice,
        # so refuse: the docs say to delete the other copy.
        _LOG.warning("h3_motion_context: another H3-Motion-Context copy has "
                     "already patched PackedLayout.__init__; DELETE every "
                     "other copy and restart ComfyUI.")
        return False
    _orig_init = current
    try:
        _self_test()
    except Exception as exc:
        _orig_init = None
        dups = _find_dup_installs()
        where = ""
        if dups:
            where = ("\n  Found other H3-Motion-Context copies in custom_nodes "
                     "(delete ALL of them and restart ComfyUI):\n    %s"
                     % "\n    ".join(dups))
        else:
            where = ("\n  No other H3-Motion-Context folder found in custom_nodes; "
                     "another copy may be loaded from elsewhere (a zip, a venv, "
                     "or a stale __pycache__). Search your whole ComfyUI "
                     "directory for folders named *H3-Motion-Context* and delete "
                     "every copy EXCEPT this fork (ComfyUI-H3-Motion-Context-"
                     "MultiRef), then restart ComfyUI.")
        _LOG.warning("h3_motion_context: self-test failed (%s), patch not "
                     "applied. Interior keyframe anchors unavailable. This is "
                     "almost always caused by a SECOND copy of the H3-Motion-"
                     "Context custom node (the upstream package or an older "
                     "version of this fork) being installed at the same time, "
                     "so both patches fight over PackedLayout.__init__ and "
                     "double-wrap it.%s", exc, where)
        return False
    _patched_init._h3mc_orig_init = _orig_init
    _patched_init._h3mc_layout_patcher = True
    mm.PackedLayout.__init__ = _patched_init
    _applied = True
    _LOG.info("h3_motion_context: interior keyframe anchors enabled")
    return True
