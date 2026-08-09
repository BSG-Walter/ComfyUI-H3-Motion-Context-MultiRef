# H3 Motion Context — multi-ref compatibility fork

> **Modified fork, 2026-08-09.** Original project by [NikoDemon80](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context), licensed GPL-3.0.

This fork keeps H3 Motion Context's clip-chaining behavior and adds one compatibility fix: **ordinary MiniMax H3 Ref2VA refs can coexist with Motion Context timeline audio**.

## What changed

Upstream timeline audio used to replace/assume the entire `minimax_refs` list. In Ref2VA graphs that can discard character/image refs or make layout placement reject multiple refs.

This fork:

- preserves existing Ref2VA refs;
- appends the Motion Context audio ref last;
- shifts only that marked audio-ref block onto the continuation timeline;
- self-tests the case `2 image refs + 1 Motion Context audio ref` at startup.

See [MODIFICATIONS.md](MODIFICATIONS.md) for the exact behavior and [patches/multi_ref_timeline_audio.patch](patches/multi_ref_timeline_audio.patch) for the source diff.

## Install

Put this folder in:

```text
ComfyUI/custom_nodes/ComfyUI-H3-Motion-Context/
```

Restart ComfyUI. Normal startup should include:

```text
h3_motion_context: interior keyframe anchors enabled
h3_motion_context: keyframe/ref coexistence enabled
```

If a layout assumption changes upstream, the runtime self-test is designed to fail loudly instead of silently moving the wrong rows.

## Recommended Motion Context settings

For the tested continuation setup:

```text
context_length:       22
encode_mode:          video
anchor_mode:          head
audio_mode:           timeline
audio_context_length: 22
```

Use the Trim node on the duplicated head before stitching.

## Ref2VA + timeline audio

A graph may now contain ordinary Ref2VA refs before the Motion Context timeline-audio ref. The Motion Context audio ref must remain **last**; `nodes.py` enforces that ordering by appending it after normal conditioning.

Example conceptual order:

```text
Ref2VA image ref
Ref2VA image ref
(optional other ordinary refs)
Motion Context timeline-audio ref   <- appended last
```

## Example workflow

`example_workflows/Simple Motion Context - No Reference Images.json` is a clean 6-clip Motion Context chain with a T2V start, one active continuation, four optional sequential continuations, 22-frame visual context, and 22-frame timeline audio context from the preceding joint AV latent.

`example_workflows/Advanced Motion Context - Reference Images.json` adds Ref2VA character references with 39-frame visual and 39-frame timeline-audio Motion Context and demonstrates the multi-ref compatibility patch.

`example_workflows/Music Video Motion Context - Reference Images + Song.json` is a 15-slot Ref2VA music-video/lip-sync chain using 22-frame visual-only Motion Context plus matching original-song slices for each clip.

All three example workflows embed their matching Director Prompt note above the first clip prompt area.

It is intentionally a basic Motion Context example and does not itself use Ref2VA refs, so the MultiRef compatibility patch is not required for this particular workflow. It is kept as the stock-compatible baseline for later workflow variants.

## Important limitation

This patch is specifically for **Ref2VA refs + Motion Context timeline audio** coexistence. It does not make long recursive generated-audio chains lossless. For fixed-song lip-sync workflows, using the original song slices as the audio reference and Motion Context for video only may be preferable.

## License / upstream

Original project and copyright: **NikoDemon80**. This modified version remains under **GPL-3.0**. See [LICENSE](LICENSE).

Upstream repository: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context
