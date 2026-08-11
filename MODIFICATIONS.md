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
2. **`patch_payload.py`** wraps `MiniMaxH3Model._cond_audio_rows` with `_patched_cond_audio_rows`: it re-runs the stock packing loop and, per marked ref block, overrides the noise-aug strength (`aug = ref.get(MC_AUDIO_STRENGTH, default_aug)`, same seed stream as stock). The timestep mapping in `_forward` stays untouched (global `audio_cond_noise_aug`).
3. **`nodes.py`** parses `"strengths"` from `audio_state`, clamps each to `0.05..1.0`, and rides the value on the ref dict. `_ensure_h3_runtime_patches()` now installs three patches (layout, extra_conds, cond-audio rows).
4. **`js/h3_custom_audio.js`** adds per-slot `audio N strength` number widgets (non-serialized, kept in `audio_state`).

All three runtime patches (layout, extra_conds, cond-audio) support adoption/takeover so the fork can coexist with the upstream `ComfyUI-H3-Motion-Context` package, which otherwise double-wraps `PackedLayout.__init__` and breaks the self-test.

## Continuous per-clip strength blend (2026-08-10)

Replaces the noise-aug override with a true influence blend: strength is now a continuous mix between the reference clip and what the model itself is generating at that spot of the clip, per denoise step.

- `s = 1.0` pins the clip exactly (stock rows); `s = 0.5` is an even mix; `s = 0.1` is a light hint; the model keeps regenerating its own audio under the block, so low strengths no longer degrade into fixed noise.
- **`patch_layout.py`** rides `motion_context_audio_strength` (clamped `0.05..1.0`) on the ref dict, untouched by the layout patch (identical `position_ids` with or without the key).
- **`patch_payload.py`** adds a fourth runtime patch wrapping `MiniMaxH3Model.forward` (`apply_forward_patch`): once per sampling run it maps each weak block's steps onto the clip's own target audio steps via `position_ids` times (`_audio_blend_map`), and every forward call stashes the evolving target rows at those steps (`pack_audio(x[1])`, channel-sequential to match the block's row order) into `_BLEND_ROWS_KEY`.
- `_patched_cond_audio_rows` now mixes `s * clip_rows + (1 - s) * stashed_target_rows` per weak block; unmarked refs keep the stock global `audio_cond_noise_aug` path exactly. A row-count mismatch falls back to stock with a warning.
- **`nodes.py`** installs the forward patch as the fourth entry in `_ensure_h3_runtime_patches()`.
- All four runtime patches support adoption/takeover, with `MODIFICATIONS.md` docs and Troubleshooting keeping the deletion guidance current.

## Example workflow set

The upstream example workflows are not included in this fork bundle. The examples folder uses a custom workflow family instead, beginning with `Simple Motion Context - No Reference Images.json` as the stock-compatible baseline. Its layout is intentionally retained as the visual template for later variants.

The example set also includes `Advanced Motion Context - Reference Images.json`, which demonstrates the patched Ref2VA + Motion Context timeline-audio coexistence path. It intentionally follows the same visual layout baseline as the simple workflow while retaining the extra global-reference and experimental full-audio-reference sections.

The custom example family also includes `Music Video Motion Context - Reference Images + Song.json`, a 15-slot visual-only Motion Context music-video template. It uses exact original-song slices as Ref2VA audio references while keeping recursive Motion Context audio disabled for long-run lip-sync stability.

Each custom example workflow now embeds its matching Director Prompt note above Clip 1 so the prompting rules travel with the workflow itself.
