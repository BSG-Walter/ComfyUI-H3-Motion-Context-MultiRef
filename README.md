# H3 Motion Context — MultiRef + Custom Keyframes

> **Modified fork.** Original project by [NikoDemon80](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context), licensed GPL-3.0.

H3 Motion Context for MiniMax H3, with MultiRef compatibility and arbitrary-position visual keyframes.

## Fork additions — 2026-08-10

- **H3 Custom Keyframes** — place still-image keyframes at arbitrary frame positions in the generated video.
- **H3 Custom Audio** — pin audio clips at arbitrary frame positions of the target clip's audio timeline (beginning, middle or end), the audio counterpart of Custom Keyframes.
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

## H3 Custom Audio

The audio counterpart of H3 Custom Keyframes: pin audio clips at arbitrary positions of the target clip's own audio timeline, so the model hears them as established sound and generates the rest around them.

The node starts with one audio slot and lets you add or remove more with **+ Add audio** and **- Remove audio**. Each slot takes an `AUDIO` clip and a frame position, plus two node-level controls:

```text
indexing:  1-based / 0-based frame positions
align:     end    - the clip finishes at the chosen frame; the model continues from it
           start  - the clip begins at the chosen frame; the model leads into it
```

Examples (1-based, align = end):

```text
AUDIO 1 -> 1      opening sound over the first frame
AUDIO 2 -> 79     sound leading into frame 79 (middle injection)
AUDIO 3 -> 22     one clip ending where the next begins: contiguous bed
```

Unused media is windowed at the anchor: an `end`-aligned clip longer than its position is tail-sliced so it always finishes exactly at the chosen frame; a `start`-aligned clip is head-sliced so it never runs past the last frame. Duplicate end frames and out-of-range positions are rejected. Blocks are appended to any existing `minimax_refs`, so they coexist with Ref2VA refs and Motion Context timeline audio.

### Per-clip strength

Each audio slot gets a **strength** widget (`audio N strength`) that sets the clip's **influence** on its zone continuously:

```text
1.0      the clip is pinned exactly (default) - the model reproduces it faithfully
0.9      almost the clip - the model may only reshape minor details
0.5      half clip, half the model's own generation
0.1      a light hint - the model creates most of the sound
  ~0     transparent - the zone is essentially free
```

Every denoising step blends the clip with what the model itself is generating at that zone (`strength * clip + (1 - strength) * generation`), so intermediate values behave like true fractions of influence with no cutoffs and no noise floor. The strength lives per audio slot inside `audio_state` (the `"strengths"` array), so it travels with the widget state and survives copies.

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

## Troubleshooting

**`self-test failed (found 4 rows in marked audio ref slot ...)` on startup or first run** means a **second copy of the H3-Motion-Context custom node is installed** (the upstream `ComfyUI-H3-Motion-Context` package or an older version of this fork). Both install the same MiniMax H3 runtime patch, and the second application double-wraps `PackedLayout.__init__`, so the self-test finds half the expected rows. **Delete every other H3-Motion-Context folder** from `custom_nodes` (keep only `ComfyUI-H3-Motion-Context-MultiRef`), clear `__pycache__` if one lingers, and restart ComfyUI. When this happens the console now prints a message listing the detected duplicate folders (or a search hint if none is found). As a safety net this fork also takes over from another `h3_motion_context` wrapper it can recognize at install time, but the duplicate install should be removed anyway since this fork replaces the upstream package entirely.

## License / upstream

Original project and copyright: **NikoDemon80**. This modified version remains under **GPL-3.0**. See [LICENSE](LICENSE).

Upstream repository: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context
