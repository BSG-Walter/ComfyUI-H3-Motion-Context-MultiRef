# H3 Motion Context — MultiRef + Custom Keyframes

> **Modified fork.** Original project by [NikoDemon80](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context), licensed GPL-3.0.

H3 Motion Context for MiniMax H3, with MultiRef compatibility and arbitrary-position visual keyframes.

## Fork additions — 2026-08-10

- **H3 Custom Keyframes** — place still-image keyframes at arbitrary frame positions in the generated video.
- **Lazy runtime patches** — the H3 compatibility patches are installed on first use rather than at ComfyUI startup.

See [MODIFICATIONS.md](MODIFICATIONS.md) for details on the MultiRef changes.

## Install

Clone the repository into your ComfyUI `custom_nodes` directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef.git
```

Then restart ComfyUI and hard-refresh the browser.

## H3 Custom Keyframes

The node starts with three image keyframes and lets you add or remove more with **+ Add keyframe** and **- Remove keyframe**.

With `indexing = 1-based`, positions can be placed anywhere in the target timeline, for example:

```text
KF1 -> 1
KF2 -> 22
KF3 -> 79
KF4 -> 122
KF5 -> 362
```

The node sorts anchors by frame position, rejects duplicate or out-of-range positions, and VAE-encodes each image as H3 conditioning at that point in the timeline.

When using **H3 Custom Keyframes**, put all visual keyframes on this node, including first/last-frame anchors if you want them.

These are conditioning anchors rather than deterministic morph points; H3 still generates the motion and transitions between them from the prompt.

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

A graph may contain ordinary Ref2VA refs before the Motion Context timeline-audio ref. The Motion Context audio ref is appended after the existing refs.

Example conceptual order:

```text
Ref2VA image ref
Ref2VA image ref
(optional other ordinary refs)
Motion Context timeline-audio ref
```

## Example workflows

`example_workflows/Simple Motion Context - No Reference Images.json` is a clean 6-clip Motion Context chain with a T2V start, one active continuation, four optional sequential continuations, 22-frame visual context, and 22-frame timeline audio context from the preceding joint AV latent.

`example_workflows/Advanced Motion Context - Reference Images.json` adds Ref2VA character references with 39-frame visual and 39-frame timeline-audio Motion Context and demonstrates the MultiRef compatibility patch.

`example_workflows/Music Video Motion Context - Reference Images + Song Driven Lipsync.json` is a 15-slot Ref2VA music-video/lip-sync chain using 22-frame visual-only Motion Context plus matching original-song slices for each clip.

`example_workflows/Custom Keyframes Example.json` demonstrates H3 Custom Keyframes with three still-image anchors.

## Important limitation

The MultiRef compatibility patch is specifically for **Ref2VA refs + Motion Context timeline audio** coexistence. It does not make long recursive generated-audio chains lossless. For fixed-song lip-sync workflows, using the original song slices as the audio reference and Motion Context for video only may be preferable.

## License / upstream

Original project and copyright: **NikoDemon80**. This modified version remains under **GPL-3.0**. See [LICENSE](LICENSE).

Upstream repository: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context
