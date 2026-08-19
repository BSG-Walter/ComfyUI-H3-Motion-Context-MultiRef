# ComfyUI H3 BSG Timeline Editor

A standalone, non-linear video-editor timeline node for **MiniMax H3** (Hailuo-01 / H3) in ComfyUI.

The **H3 BSG Timeline Editor** (`MiniMaxH3Timeline`) brings a full multi-track visual timeline canvas directly into your ComfyUI workflow. Arrange still images, video clips, and audio tracks along a frame-accurate ruler, trim and split clips, edit retention strength envelopes, preview video and audio in real time, and pass exact inpainting guidance straight to standard `KSampler`.

Powered **100% natively** by ComfyUI core MiniMax H3 keyframe guides (`minimax_keyframes`) and native per-token denoise masks (`noise_mask` on `NestedTensor`). No monkey-patches, no runtime layout overrides, and no custom sampler hacks.

---

## Key Features

- **Full NLE Timeline Canvas**: 840px interactive multi-track canvas with dedicated Video/Image (top) and Audio (bottom) tracks, customizable frame ruler, zoom controls, and an interactive playhead.
- **Native Dual-Guidance Control**:
  - **Cross-Attention Conditioning**: Attaches core `minimax_keyframes` guidance to the prompt conditioning.
  - **Per-Token Denoise-Mask Inpainting**: Encodes clips into a clean composite latent with an exact per-token `noise_mask`. When sampled in standard `KSampler` with `denoise: 1.0`, masked regions (strength 1.0) are preserved with 100% fidelity while unmasked frames are smoothly synthesized.
- **Direct Drag & Drop / File Uploads**: Upload images (PNG, JPG, WebP, GIF), videos (MP4, MOV, WebM, MKV, AVI), or audio files (WAV, MP3, FLAC, M4A, OGG) straight into the node. They decode directly in the browser with video thumbnail scrubbing and 128-bucket audio waveforms.
- **Linked & Unlinked Audio**: Video clips automatically load their embedded audio track. Keep audio linked to move and trim it synchronously with the video, or unlink (`🔗`) to offset, stretch, or independently edit/delete the audio track.
- **Retention Strength & Envelopes**: Control how strongly the model adheres to clip content (0.0 to 1.0). Drag the strength bar or double-click to place keyframe envelope points for smooth cross-fades and transitions. Video clip envelopes visualize stepped boundaries matching the model's 17-frame token grid.
- **Clip Slicing & Precision Trimming**:
  - Drag the left/right edges of clips to trim their length and adjust the source playback window (`src_start`).
  - Move the playhead and hit `✂ Split` (or press `S`) to slice a clip into two independent segments.
- **Multi-Selection & Clipboard**: Select multiple clips with `Ctrl`/`Cmd` + click, drag groups together while preserving relative spacing, and copy/paste (`Ctrl+C` / `Ctrl+V`) into the next available gap.
- **Dedicated History (Undo / Redo)**: Step backward (`↶ Undo`) and forward (`↷ Redo`) through clip manipulations without conflicting with ComfyUI's global graph undo.
- **Real-Time Playback & Scrubbing**: Scrub across the ruler or press `Space` / `▶ Play` to play synchronized video frames and audio in real time via WebAudio.
- **Smart Magnet Snapping (`🧲`)**: Snaps clip edges to adjacent clips, to the playhead, and to the MiniMax H3 17-frame VAE grid lines.
- **State Export / Import**: Easily save timeline configurations to JSON files or load existing project timelines.

---

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/BSG-Walter/ComfyUI-H3-Motion-Context-Timeline.git
```

Make sure the following Python dependencies are installed in your ComfyUI environment (standard in most ComfyUI setups):

```bash
pip install av torch torchvision torchaudio pillow numpy
```

Restart ComfyUI and hard-refresh your browser (`Ctrl+F5` or `Cmd+Shift+R`).

---

## Node Anatomy

```text
               +-------------------------------------------+
               |            H3 BSG Timeline Editor             |
               +-------------------------------------------+
[CONDITIONING] | conditioning                 conditioning | [CONDITIONING]
         [VAE] | video vae                          latent | [LATENT]
         [VAE] | audio vae                                 |
      [LATENT] | latent                                    |
      [STRING] | timeline_state (UI)                       |
     [COMBOBOX]| crop (disabled / center)                  |
        [INT]  | fps (default: 24)                         |
        [INT]  | total_frames (default: 243)               |
               +-------------------------------------------+
               |   [Visual BSG Timeline Editor Canvas 840px]   |
               +-------------------------------------------+
```

### Inputs

| Input | Type | Description |
| :--- | :--- | :--- |
| **`conditioning`** *(required)* | `CONDITIONING` | Positive conditioning / prompt from `CLIPTextEncode`. Timeline keyframe guides are attached to this stream. |
| **`video vae`** *(required)* | `VAE` | MiniMax H3 Video VAE used to encode still images and video clips into the 24-channel latent space. |
| **`audio vae`** *(required)* | `VAE` | MiniMax H3 Audio VAE used to encode audio tracks into the 32-channel stereo latent space. |
| **`latent`** *(required)* | `LATENT` | Empty or initial MiniMax H3 AV latent (e.g. from `EmptyMiniMaxH3LatentAV`). Sets the generation resolution (width/height) and duration. |
| **`crop`** | `COMBO` | Scaling behavior when source images/videos do not match target latent aspect ratio: `disabled` (stretch) or `center` (crop to center). |
| **`fps`** *(optional)* | `INT` | Frame rate for video playback and audio synchronization (default: `24`, range: `1 - 240`). |
| **`total_frames`** *(optional)* | `INT` | Ruler length in frames (default: `243`, step: `17`). Automatically snaps to valid MiniMax H3 lengths (`5 + 17k`: 5, 22, 39, 56, 73, 243...). |
| **`image_N`, `video_N`, `audio_N`** | `DYNAMIC` | Optional socket inputs to wire image/video/audio tensors from other ComfyUI nodes instead of uploading files. |

### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| **`conditioning`** | `CONDITIONING` | Conditioning payload containing the compiled `minimax_keyframes` list. |
| **`latent`** | `LATENT` | A `NestedTensor` containing the composite clean video/audio latents (`samples`) and per-token denoise masks (`noise_mask`). Connect directly to `KSampler`. |

---

## How It Works: Flow Matching & Inpainting

MiniMax H3 uses a temporal DiT architecture where:
- **17 pixel frames** correspond to **5 latent tokens** on a `(1, 4, 4, 4, 4)` temporal grid.
- Video latents are 24-channel tensors (`[B, 24, T_v, H/16, W/16]`).
- Audio latents are 32-channel tensors (`[B, 32, 2, T_a]`).

### Denoise-Mask Mechanism
When you place a clip on the timeline:
1. **VAE Encoding**: The clip content is aligned to the 17-frame chunk boundary and encoded through the video/audio VAE.
2. **Latent Composition**: The encoded tokens are placed into `clean_video` and `clean_audio` tensors.
3. **Noise Mask Construction**: A per-token `noise_mask` is generated where:
   $$\text{mask\_value} = 1.0 - \text{strength}$$
   - **`strength = 1.0` $\rightarrow$ `mask = 0.0`**: The region is pinned clean. The sampler preserves this content identically.
   - **`strength = 0.0` $\rightarrow$ `mask = 1.0`**: The region is completely open for pure text-to-video generation.
   - **`0.0 < strength < 1.0`**: The region receives soft guidance with intermediate noise scheduling.
4. **Sampling**: When connected to `KSampler` with `denoise: 1.0`, the sampler uses the native inpaint pipeline to seamlessly merge generated motion around the locked clip frames.

---

## Using the BSG Timeline Editor

### 1. Adding Media

You can populate the timeline using two methods:

- **Direct File Upload (Recommended)**:
  - Click the **`+ image`**, **`+ video`**, or **`+ audio`** buttons beneath the timeline.
  - Or **right-click** any empty space on the canvas and choose `+ Insert Image`, `+ Insert Video`, or `+ Insert Audio`.
  - Pick a file from your computer. The file is uploaded to ComfyUI's input directory and immediately decoded on the canvas.
- **Node Input Sockets**:
  - Connect standard `IMAGE` or `AUDIO` nodes to the dynamic `image_N`, `video_N`, `video_audio_N`, or `audio_N` sockets.

### 2. Supported Clip Types

- **Still Images (`image`)**: Stills (PNG, JPG, WebP) or animated GIFs/WebP. Stills default to a length of 22 frames and can be stretched or trimmed to any duration.
- **Video Clips (`video`)**: Video files (MP4, MOV, WebM, etc.). Automatically loads video frames and associated audio.
- **Audio Clips (`audio`)**: Audio files (WAV, MP3, FLAC, AAC, etc.). Visualized with a detailed waveform on the audio lane.

### 3. Timeline Tracks & Lanes

- **Top Lane (Video Track)**: Houses still images and video clips.
- **Bottom Lane (Audio Track)**: Houses standalone audio clips and unlinked audio tracks from video clips.

### 4. Clip Editing & Manipulation

- **Move**: Click and drag any clip horizontally.
- **Trim Left / Right**: Grab the left or right edge of a clip and drag:
  - Dragging the **right edge** adjusts the duration (`len`).
  - Dragging the **left edge** changes the starting frame while offsetting the internal source window (`src_start`), keeping the cut point intuitive.
- **Link / Unlink Audio (`🔗`)**:
  - By default, a video's audio moves and trims synchronously with the video block.
  - Click the **`🔗`** link icon or use the context menu to **unlink audio**.
  - Once unlinked, the audio band appears as a separate block on the audio lane, allowing you to offset sound timing, trim audio independently, or delete the audio track entirely.
- **Split Clip (`✂ Split` / `S`)**:
  - Position the playhead over a clip and click `✂ Split` on the toolbar or press `S`.
  - The clip splits into two independent clips with preserved source frame offsets (`src_start`).

### 5. Strength & Envelopes

Every clip features an adjustable green retention curve:
- **Flat Strength**: Hover over the horizontal green line inside a clip (cursor changes to `ns-resize`) and drag up/down to adjust strength between `0.00` and `1.00`.
- **Envelope Keyframes**:
  - **Double-click** on the green line to add a keyframe point.
  - Drag points up or down to create fades, ramps, and custom blends.
  - **Right-click** an envelope point to remove it.
  - Video envelopes automatically snap to token boundaries for exact DiT synchronization.

### 6. Playback & Scrubbing

- **Scrubbing**: Click or drag across the top ruler to move the red playhead. Video clips automatically seek to preview the exact frame under the playhead.
- **Real-Time Playback**: Click **`▶ Play`** on the toolbar or press **`Space`** to preview your video edits and hear synchronized audio.
- **Loop & Bounds**: Playback automatically respects the total timeline duration (`total_frames`).

### 7. Zoom, Pan & Ruler Controls

- **Zoom Slider & Buttons**: Use the log-scale slider or `+` / `−` buttons to zoom from full project overview down to individual frame detail.
- **Panning Scrollbar**: When zoomed in, drag the horizontal scrollbar located directly beneath the timeline canvas.
- **Unit Toggle (`F` / `S`)**: Click the unit button on the toolbar to switch the ruler display between **Frames** (`frame 1`, `frame 22`...) and **Seconds** (`0.0s`, `1.5s`...).

---

## Keyboard Shortcuts & Toolbar Reference

| Shortcut / Button | Action |
| :--- | :--- |
| **`Space`** / `▶ Play` / `⏹ Stop` | Toggle timeline preview playback (video frames + audio). |
| **`S`** / `✂ Split` | Split the clip under the playhead into two segments. |
| **`Ctrl + C`** / `Cmd + C` | Copy selected clip(s). |
| **`Ctrl + V`** / `Cmd + V` | Paste copied clip(s) into the next available gap. |
| **`Delete`** / `Backspace` | Delete selected clip(s). |
| **`Ctrl + Click`** / `Cmd + Click` | Multi-select clips for group movement or deletion. |
| **`↶ Undo`** | Undo the last timeline action (move, trim, split, envelope edit). |
| **`↷ Redo`** | Redo the previously undone action. |
| **`🧲 Snap`** | Toggle magnet snapping on/off. |
| **`F` / `S`** | Toggle ruler units between Frames and Seconds. |
| **`+` / `−`** | Zoom in / zoom out horizontally. |
| **`🗑 Clear`** | Clear all clips from the timeline canvas. |
| **`⤓ Export`** | Export current timeline layout to a JSON file. |
| **`⤒ Import`** | Import a timeline layout from a JSON file. |
| **Right-Click (Clip)** | Open context menu (Copy, Delete, Replace Media, Delete Audio). |
| **Right-Click (Empty Lane)** | Open insertion menu (Insert Image, Insert Video, Insert Audio, Paste). |

---

## Standard Workflow Integration

Below is the recommended standard setup for MiniMax H3 video generation with the BSG Timeline Editor:

```text
[ CLIPTextEncode (Prompt) ] --------> conditioning ──> [ H3 BSG Timeline Editor ] ──> conditioning ──> [ KSampler ]
[ EmptyMiniMaxH3LatentAV ] ---------> latent ─────���──> [                    ] ──> latent ────────> (denoise: 1.0)
[ VAELoader (H3 Video VAE) ] -------> video vae ─────> [                    ]                          |
[ VAELoader (H3 Audio VAE) ] -------> audio vae ─────> [                    ]                          v
                                                                                                [ VAEDecode ]
                                                                                                       |
                                                                                                       v
                                                                                           [ VHS_VideoCombine / Save ]
```

### Crucial Setting: `denoise = 1.0` in KSampler
Always set **`denoise: 1.0`** on your `KSampler`. 
Because the `H3 BSG Timeline Editor` produces exact per-token `noise_mask` maps, the sampler uses full denoise to synthesize unmasked areas while the native inpainting engine automatically protects and locks your timeline clips according to their retention strength.

---

## Troubleshooting & Best Practices

- **Frame Count Snapping (`total_frames`)**:
  MiniMax H3 VAE requires video lengths to follow the formula $5 + 17k$ (e.g. 5, 22, 39, 56, 73, 90, 107, 124, 141, 158, 175, 192, 209, 226, 243...). The `total_frames` widget automatically snaps to these valid increments.
- **Audio Sample Rates**:
  The timeline automatically resamples uploaded audio waveforms to match the Audio VAE's native sample rate (typically 32,000 Hz or 16,000 Hz). Ensure `torchaudio` is installed for high-quality resampling.
- **Clips Extending Past the Ruler**:
  Any clips placed beyond the ruler duration or latent boundary are dimmed on the canvas and safely clamped by the backend without raising errors.
- **Aspect Ratio & Cropping**:
  If using images or videos with different resolutions, set `crop: center` to center-crop inputs to the target latent resolution, or `crop: disabled` to rescale them.

---

## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. See the [LICENSE](LICENSE) file for full details.
