# H3 Motion Context — Timeline

H3 Motion Context Timeline for MiniMax H3, with a full **video-editor timeline** widget that lets you place still images, video clips and audio clips at any frame position on a visual canvas.

Powered natively by ComfyUI core MiniMax H3 guides (`minimax_keyframes`) without any monkey-patches or runtime layout overrides. Fully compatible with other custom nodes.

## What this node provides

### H3 Timeline Editor
The **H3 Timeline Editor** node (`MiniMaxH3Timeline`) gives you a visual canvas where you arrange everything on a single timeline.

- **Canvas timeline** (840px wide) with two lanes (video above, audio below), a frame ruler, zoom controls and a playhead.
- **Undo & Redo buttons** — dedicated `↶ Undo` and `↷ Redo` toolbar buttons to safely step backward and forward through edits (clip moves, trims, deletions, paste, envelope tweaks) without interfering with ComfyUI's global graph undo.
- **Multi-selection & Group drag** — select multiple clips with `Ctrl`/`Cmd` + click and drag them together across the timeline while preserving their relative spacing.
- **Copy & Paste** — copy and paste selected clips using standard shortcuts (`Ctrl+C` / `Ctrl+V`) or the context menu. Pasting scans occupied lanes and automatically places clips in the next available free space without overlapping.
- **Context menus** — right-click on any clip or empty space in the video/audio lanes to quickly insert media (`+ Insert Image`, `+ Insert Video`, `+ Insert Audio`), paste clips, replace media, edit envelope points, or delete.
- **Drag-and-drop clips** — still images, video clips and audio clips can be placed at any frame position. Clips can extend past the ruler end-line (a visual delimiter with snapping, like the playhead) and dim automatically when they do.
- **Per-clip source trim** — drag the left/right edges of a clip to set its length and source start. Linked video/audio drags as one unit; unlinked audio can be moved independently.
- **Snapping** — clips snap to each other, to the playhead and to the end-line. Toggle with the magnet button.
- **Zoom slider** — log-scale zoom from the ruler overview down to per-frame detail, plus the standard +/− buttons.
- **Playable preview** — the playhead plays through the timeline, syncing video thumbnails; playback stops at the end-line.
- **File-backed clips** — upload videos/images/audio directly into the timeline and they decode on the node (video thumbnails + audio waveform). No need to wire separate LoadImage / VHS nodes when you just want to drop a file.
- **Out-of-range tolerance** — clips placed beyond the latent's frame count are clamped-and-warned by the backend (parked at the last frame), never fatal. Audio clips parked or trimmed log a warning instead of raising.

## Install

Clone the repository into your ComfyUI `custom_nodes` directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/BSG-Walter/ComfyUI-H3-Motion-Context-Timeline.git
```

Then restart ComfyUI and hard-refresh the browser.

## H3 Timeline Editor

The `H3 Timeline Editor` node gives you one ordered list of clips, each pinned at a 1-based start frame on a visual canvas:

```text
image  a still, pinned at its frame (an H3 custom keyframe)
video  a full clip; every latent step is pinned at its own frame
audio  a window of sound pinned on the audio track
```

A video's audio is linked by default: it follows the clip's position and length. Unlink to move or trim it independently (`audio_start` / `audio_len` / `audio_align`). Audio windows are cut from the head (`head`) or the tail (`tail`) of their source.

**fps** and **total_frames** are widgets you can edit by hand or convert to input slots (right-click → Convert to input) and drive from other nodes. The timeline ruler length follows `total_frames`; the generated frame count still comes from the wired latent (clips past the ruler end-line are dimmed on the canvas and clamped-and-warned at the backend).

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

**`self-test failed (found 4 rows in marked audio ref slot ...)` on startup or first run** means a **second copy of the H3-Motion-Context custom node is installed** (the upstream `ComfyUI-H3-Motion-Context`, the `ComfyUI-H3-Motion-Context-MultiRef` fork, or an older version of this fork). Both install the same MiniMax H3 runtime patch, and the second application double-wraps `PackedLayout.__init__`, so the self-test finds half the expected rows. **Delete every other H3-Motion-Context folder** from `custom_nodes` (keep only this one), clear `__pycache__` if one lingers, and restart ComfyUI. When this happens the console now prints a message listing the detected duplicate folders (or a search hint if none is found). As a safety net this fork also takes over from another `h3_motion_context` wrapper it can recognize at install time, but the duplicate install should be removed anyway since this fork replaces the upstream package entirely.

## License / upstream

Original project and copyright: **NikoDemon80**. This modified version remains under **GPL-3.0**. See [LICENSE](LICENSE).

Upstream repository: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context
