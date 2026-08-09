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

## Example workflow set

The upstream example workflows are not included in this fork bundle. The examples folder uses a custom workflow family instead, beginning with `Simple Motion Context - No Reference Images.json` as the stock-compatible baseline. Its layout is intentionally retained as the visual template for later variants.

The example set also includes `Advanced Motion Context - Reference Images.json`, which demonstrates the patched Ref2VA + Motion Context timeline-audio coexistence path. It intentionally follows the same visual layout baseline as the simple workflow while retaining the extra global-reference and experimental full-audio-reference sections.

The custom example family also includes `Music Video Motion Context - Reference Images + Song.json`, a 15-slot visual-only Motion Context music-video template. It uses exact original-song slices as Ref2VA audio references while keeping recursive Motion Context audio disabled for long-run lip-sync stability.

Each custom example workflow now embeds its matching Director Prompt note above Clip 1 so the prompting rules travel with the workflow itself.
