# Modified fork notice

This repository is a modified version of **NikoDemon80/ComfyUI-H3-Motion-Context**.

- Upstream: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context
- Original author: NikoDemon80
- License: GPL-3.0
- Modification date: 2026-08-09

## Multi-ref timeline-audio compatibility change

Upstream Motion Context can combine H3 keyframes and refs, but its timeline-audio path assumes the Motion Context audio ref is the only `minimax_refs` block. That conflicts with Ref2VA graphs that already contain ordinary image/video/audio refs.

This fork changes two places:

1. **`nodes.py`** preserves existing `minimax_refs` and appends the special Motion Context audio ref last via `conditioning_set_values(..., append=True)`.
2. **`patch_layout.py`** locates the marked Motion Context audio ref's actual stock coordinate slot after any ordinary refs, then shifts only that block onto the continuation timeline.

The marked Motion Context audio ref is intentionally required to be the **last ref block**. The runtime self-test additionally covers two ordinary image refs followed by one marked Motion Context timeline-audio ref.

No ComfyUI source files are modified on disk; this remains a runtime patch custom node.

## Per-clip audio strength (2026-08-10)

Adds an optional per-audio-slot strength control on top of the core's global `audio_cond_noise_aug` mechanism.

1. **`patch_layout.py`** defines the ref payload key `motion_context_audio_strength`.
2. **`patch_payload.py`** wraps `MiniMaxH3Model._cond_audio_rows` with `_patched_cond_audio_rows` (marked rows pinned exact, unmarked stock) and the forward wrapper flips weak audio blocks out of the layout on the audio schedule (`t_a`).
3. **`nodes.py`** parses `"strengths"` from `audio_state`, clamps each to `0.05..1.0`, and rides the value on the ref dict. `_ensure_h3_runtime_patches()` now installs the layout, extra_conds, cond-audio rows, cond-video rows and forward patches.
4. **`js/h3_custom_audio.js`** adds per-slot `audio N strength` number widgets (non-serialized, kept in `audio_state`).

All three runtime patches (layout, extra_conds, cond-audio) support adoption/takeover so the fork can coexist with the upstream `ComfyUI-H3-Motion-Context` package, which otherwise double-wraps `PackedLayout.__init__` and breaks the self-test.

## Per-clip strength (2026-08-11)

Strength is a **pin-then-flip schedule**, for both modalities. The model
learns how noisy a condition is from the timestep assigned to its tokens
(`seg_t["cond"] = max(t_v, visual_cond_noise_aug)` and
`seg_t["ref_audio"] = max(t_a, audio_cond_noise_aug)` in `_forward`), so
custom-strength rows mixed under the stock claim are a lie and the noise
bakes into the output (observed: video grain below ~0.75 with fixed
noise-aug, audio noise on low-denoise runs). Instead a marked weak block
is pinned EXACT while its timeline's progress stays below its strength
(clean rows, claim forced to 0.999 - the canonical image-to-video pair),
then its tokens are dropped from the layout for the rest of the run, so
the model's own stream covers the region with no reference at all.
Video progress is `t_v = 1 - sigma`; audio progress is `t_a` on the
audio's own shifted schedule (`time_shift_sigma`). Strength is the
fraction of the run the block stays pinned: `1.0` exact, `0.5` pinned
half then free, `0.0` a pure prompt block. Nothing noisy is ever shown
to the model.

- `s = 1.0` pins the clip exactly; `s = 0.9` almost the clip; `s = 0.5`
  half pinned / half free; `s = 0.1` a light early-structure hint.
- **`patch_layout.py`** rides `motion_context_audio_strength` /
  `motion_context_video_strength` (clamped `0.05..1.0`) on refs/keyframes,
  untouched by the layout patch (identical `position_ids` with or without
  the keys).
- **`patch_payload.py`**:
  - `apply_cond_video_patch` / `apply_cond_audio_patch`: the row packers
    pin marked rows exact (clean, no mixing); unmarked blocks keep the
    stock global-aug path byte-for-byte.
  - `apply_forward_patch` (wraps `MiniMaxH3Model.forward`): computes
    `t_v` / `t_a` from the step's timestep, forces the claims to 0.999
    while weak blocks are pinned, records the ACTIVE SET
    (`_h3mc_active_keyframes` / `_h3mc_active_refs`) and rebuilds
    `payload["layout"]` once per set change (the stock `_forward` re-reads
    it every step via the signature check). The payload lists are never
    mutated: the patched packers read latents off the keyframe/ref dicts
    and skip inactive blocks, so rows and layout can never desync. Unmarked
    runs pass straight through untouched.
- **`nodes.py`** installs the cond-audio, cond-video and forward patches in
  `_ensure_h3_runtime_patches()`. `MiniMaxH3CustomVideo` writes
  `MC_AUDIO_STRENGTH` on each clip's audio ref so the slot's strength
  governs its audio track too.
- All runtime patches support adoption/takeover, with `MODIFICATIONS.md`
  docs and Troubleshooting keeping the deletion guidance current.

## Example workflow set

The upstream example workflows are not included in this fork bundle. The examples folder uses a custom workflow family instead, beginning with `Simple Motion Context - No Reference Images.json` as the stock-compatible baseline. Its layout is intentionally retained as the visual template for later variants.

The example set also includes `Advanced Motion Context - Reference Images.json`, which demonstrates the patched Ref2VA + Motion Context timeline-audio coexistence path. It intentionally follows the same visual layout baseline as the simple workflow while retaining the extra global-reference and experimental full-audio-reference sections.

The custom example family also includes `Music Video Motion Context - Reference Images + Song.json`, a 15-slot visual-only Motion Context music-video template. It uses exact original-song slices as Ref2VA audio references while keeping recursive Motion Context audio disabled for long-run lip-sync stability.

Each custom example workflow now embeds its matching Director Prompt note above Clip 1 so the prompting rules travel with the workflow itself.
