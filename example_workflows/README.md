# Example workflows

## Simple Motion Context - No Reference Images.json

A simple long-form MiniMax H3 Motion Context chain:

- Clip 1 starts from text-to-video.
- Clip 2 is active by default as the first Motion Context continuation.
- Clips 3-6 are included as optional sequential extensions.
- Continuations use 22 frames of visual Motion Context in `video / head` mode.
- Continuations also use 22-frame `timeline` audio context from the preceding sampled joint AV latent via `context_latent`.
- Decoded IMAGE and AUDIO tensors are concatenated directly; intermediate preview MP4s are not used as source material.
- Global clip duration and resolution controls are included.
- Global sampler and step controls drive all six clip slots.
- Optional attention / speedup nodes are present and bypassed by default.

This example does not use Ref2VA reference inputs itself; it is a clean baseline chain for Motion Context behavior.

Model filenames may need to be changed for another ComfyUI installation.

## Layout baseline

This workflow is the canonical layout baseline for the example family. Its node spacing, group positions, colors, optional-clip layout, and disabled `Speedups` group should be preserved when deriving later reference-image and patched variants unless a variant specifically requires additional nodes.

## Advanced Motion Context - Reference Images

`Advanced Motion Context - Reference Images.json`

- Ref2VA with two global character reference images on every clip.
- 39-frame visual Motion Context plus 39-frame timeline audio context.
- Requires this fork's multi-ref timeline-audio compatibility patch.
- Includes the experimental full-previous-audio reference section, muted by default.
- Uses the same clip-row spacing, Global Settings placement, final-output placement, and disabled Speedups layout as the simple baseline.
- Final output prefix: `video/motion_context`.

## Music Video Motion Context - Reference Images + Song

`Music Video Motion Context - Reference Images + Song.json`

- Ref2VA music-video / lip-sync template with two global character references.
- Up to 15 sequential clip slots; Clips 1–2 active, Clips 3–15 bypassed by default.
- 22-frame visual Motion Context only; Motion Context audio is disabled.
- Each clip receives the matching original-song slice as `<Audio 1>`.
- Final stitched picture is muxed with the original loaded song.
- Uses the same core clip spacing, Global Settings placement, final-output placement pattern, and disabled Speedups block as the simple baseline.
- The MultiRef timeline-audio patch is not required for this visual-only Motion Context variant.
- Final output prefix: `video/motion_context`.

## Embedded director prompts

Each example embeds its matching `Director Prompt for your LLM` note directly above the first clip prompt area:

- Simple workflow: no-reference-images Motion Context director.
- Advanced workflow: Ref2VA + 39/39 Motion Context director.
- Music-video workflow: 15-slot visual-only Motion Context / original-song lip-sync director.
