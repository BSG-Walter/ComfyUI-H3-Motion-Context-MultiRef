# MODIFIED FORK NOTICE: modified 2026-08-09 to allow ordinary MiniMax H3 Ref2VA refs
# to coexist with H3 Motion Context timeline-audio refs. Original project by NikoDemon80.
# See MODIFICATIONS.md. Distributed under the upstream GPL-3.0 license.

"""Pin previous-clip motion at the head of an H3 clip.

Wire it between a stock H3 conditioning node and the sampler:

    MiniMaxH3ImageToVideo (or the t2v path)
        -> H3 Motion Context
        -> guider / sampler

Two axes to test, both cheap.

encode_mode
  frames  one VAE call per frame, each pinned as its own cond block. The
          model sees N snapshots at N instants.
  video   one VAE call for the whole run. The H3 video VAE has latent_dim
          3, so it reads the batch axis as time and compresses the run
          into fewer latent steps (5 pixel frames -> 2 steps, 22 -> 7).
          Each step becomes one cond block, so the motion between frames
          lives inside the latent instead of being implied across separate
          stills. Far fewer rows and one VAE load.

anchor_mode
  head    pinned frames occupy indices 0..N-1 of the delivered timeline.
          They come back in the output, so trim that many frames off the
          front before concatenating.
  before  pinned frames sit at negative indices, ending at -1, so
          delivered frame 0 continues from them and nothing is wasted.
          Their time coordinates land below text_len, which is the range
          the text rows occupy. Whether that collision matters is exactly
          what this mode is asking.
"""

import json
import logging
import os

import av
import numpy as np
import torch
import comfy.utils
import folder_paths
import node_helpers
from comfy_extras.nodes_audio import f32_pcm
from comfy_extras.nodes_minimax_h3 import _resize
from PIL import Image, ImageOps
from safetensors.torch import load_file as _st_load, save_file as _st_save

from .patch_layout import (
    MC_KEY,
    MC_AUDIO_KEY,
    MC_AUDIO_STRENGTH,
    MC_VIDEO_STRENGTH,
    apply_patch as _apply_layout_patch,
)
from .patch_payload import (
    apply_patch as _apply_payload_patch,
    apply_cond_audio_patch as _apply_cond_audio_patch,
    apply_cond_video_patch as _apply_cond_video_patch,
    apply_forward_patch as _apply_forward_patch,
)

try:
    import torchaudio
except ImportError:
    torchaudio = None

_LOG = logging.getLogger("h3_motion_context")

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FPS = 24  # H3's native rate; audio latents run at 40 Hz, hence FRAME_RESCALE 5/3
FRAME_RESCALE = 5.0 / 3.0
AUDIO_HZ = 40.0

# Run lengths the video VAE's downscale formula max(1, (n - 5) // 17 * 5 + 2)
# actually distinguishes. Anything between two grid points encodes to the same
# number of latent steps as the lower one, but the steps then cover the FIRST
# `covered` frames of the input rather than the last: encoding 10 frames yields
# the same 2 steps as encoding 5, representing frames [-10..-6] of the source
# clip instead of [-5..-1]. The pinned run would end five frames early and the
# delivered clip would continue from the wrong instant. So off-grid requests
# are snapped DOWN before slicing, keeping content and coverage in agreement.
VIDEO_RUN_GRID = (39, 22, 5, 1)


def _ensure_h3_runtime_patches():
    """Install the H3 patches on first execution of a node that needs them."""
    if not _apply_layout_patch():
        raise RuntimeError(
            "h3_motion_context: could not enable the MiniMax H3 layout "
            "extension. Check the ComfyUI console for the self-test error; "
            "if a second H3-Motion-Context custom node (upstream package or "
            "an older version of this fork) is installed, DELETE every other "
            "copy and restart ComfyUI.")
    if not _apply_payload_patch():
        raise RuntimeError(
            "h3_motion_context: could not enable keyframe/ref coexistence. "
            "Check the ComfyUI console.")
    if not _apply_cond_audio_patch():
        raise RuntimeError(
            "h3_motion_context: could not enable per-block audio strength. "
            "Check the ComfyUI console.")
    if not _apply_cond_video_patch():
        raise RuntimeError(
            "h3_motion_context: could not enable per-block video strength. "
            "Check the ComfyUI console.")
    if not _apply_forward_patch():
        raise RuntimeError(
            "h3_motion_context: could not enable the per-keyframe pin/flip "
            "schedule. Check the ComfyUI console.")


def _pixel_frames(latent_t):
    """Pixel frames covered by latent_t latent steps."""
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))


def _step_offsets(latent_t):
    """Pixel-frame index at which each latent step begins."""
    out, acc = [], 0
    for k in range(latent_t):
        out.append(acc)
        acc += FRAME_PER_TOKEN[k % 5]
    return out


def _encode_audio_window(audio_vae, audio, seconds, tail=True):
    """Encode the last (`tail`) or first (`head`) `seconds` of a clip's audio.

    Returns ([1, 32, 2, T] latent, T) where T counts 40 Hz latent steps,
    matching what the layout calls ref_audio_t.
    """
    waveform = audio["waveform"]  # [B, C, L]
    sr = int(audio["sample_rate"])
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sr != vae_sr:
        if torchaudio is None:
            raise RuntimeError(
                "h3_motion_context: audio is %d Hz but the VAE wants %d Hz "
                "and torchaudio is not available to resample." % (sr, vae_sr))
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    want = int(round(seconds * vae_sr))
    have = int(waveform.shape[-1])
    if have < want:
        _LOG.warning("h3_motion_context: audio is %.3fs, shorter than the "
                     "%.3fs window; windowing what there is.",
                     have / vae_sr, seconds)
    elif tail:
        waveform = waveform[..., have - want:]
    else:
        waveform = waveform[..., :want]
    z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
    return z, int(z.shape[-1])


def _media_ref_path(media):
    """Resolve an uploaded media ref {name, subfolder, type} to a file path."""
    name = media.get("name") if isinstance(media, dict) else None
    if not name:
        raise ValueError("h3_motion_context: clip media ref has no name")
    sub = media.get("subfolder") or ""
    ref = "%s/%s" % (sub, name) if sub else name
    path = folder_paths.get_annotated_filepath(
        "%s [%s]" % (ref, media.get("type") or "input"))
    if not os.path.isfile(path):
        raise ValueError(
            "h3_motion_context: media file not found: %s" % ref)
    return path


def _load_image_file(media):
    """Load an uploaded still image as [1,H,W,C] float IMAGE frames."""
    img = node_helpers.pillow(
        ImageOps.exif_transpose,
        node_helpers.pillow(Image.open, _media_ref_path(media)))
    img = img.convert("RGB")
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)[None]


def _load_media_file(media):
    """Decode an uploaded audio/video file with PyAV.

    Returns {"frames": [B,H,W,C]|None, "audio": AUDIO dict|None}, decoding
    whatever streams the file actually has.
    """
    frames, audio = None, None
    with av.open(_media_ref_path(media)) as af:
        video = af.streams.video
        if video:
            collected = []
            for frame in af.decode(video[0]):
                arr = frame.to_ndarray(format="rgb24")
                collected.append(torch.from_numpy(
                    np.asarray(arr, dtype=np.float32) / 255.0))
            if collected:
                frames = torch.stack(collected)
        sounds = af.streams.audio
        if sounds:
            stream = sounds[0]
            sr = stream.codec_context.sample_rate
            n_channels = stream.channels
            chunks = []
            for frame in af.decode(stream):
                buf = torch.from_numpy(frame.to_ndarray())
                if buf.shape[0] != n_channels:
                    buf = buf.view(-1, n_channels).t()
                chunks.append(buf)
            if chunks:
                audio = {
                    "waveform": f32_pcm(torch.cat(chunks, dim=1)).unsqueeze(0),
                    "sample_rate": int(sr),
                }
    return {"frames": frames, "audio": audio}


def _slice_audio(audio, seconds):
    """Drop `seconds` of sound from the FRONT of an AUDIO dict."""
    sr = int(audio["sample_rate"])
    n = int(round(seconds * sr))
    w = audio["waveform"]
    if n > 0 and n < int(w.shape[-1]):
        return {"waveform": w[..., n:], "sample_rate": sr}
    return audio


def _streams_from_latent(latent):
    """Unpack an H3 AV latent into its contained streams.

    NestedTensor.__getitem__ broadcasts the index into every contained
    tensor rather than selecting one, so samples[0] would strip the batch
    dimension off both streams. unbind() returns the pair.
    """
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            "h3_motion_context: expected a MiniMax H3 AV latent (a nested "
            "video/audio pair), got %r" % type(samples))
    if not parts:
        raise ValueError("h3_motion_context: AV latent contains no streams")
    return parts


def _video_from_latent(latent):
    """Pull the video stream out of an H3 AV latent."""
    video = _streams_from_latent(latent)[0]
    if video.ndim == 4:  # unbatched [C,T,H,W]
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError("h3_motion_context: expected video latent [B,C,T,H,W], "
                         "got shape %s" % (tuple(video.shape),))
    return video


def _audio_tail_from_latent(latent, a_frames):
    """Slice the last `a_frames` worth of audio steps straight out of a
    generated H3 latent, skipping the decode -> re-encode round trip.

    Returns (tail latent [1, C, 2, rt], rt, overhang) where rt counts
    40 Hz latent steps and overhang is the fraction of a step by which the
    clip's audio grid extends past its last pixel frame. H3 rounds the
    audio grid UP (124 frames want 206.67 steps, the layout allocates
    207), so the latent's final step reaches ~overhang/40 s beyond the
    last frame. The decoded-audio path never sees this because match_tail
    cuts it; on this path the caller compensates the placement with it,
    so the pinned content lands exactly where its samples actually sit.
    """
    parts = _streams_from_latent(latent)
    if len(parts) < 2:
        raise ValueError(
            "h3_motion_context: context_latent has no audio stream. Wire the "
            "sampler output of an H3 AV graph, not a video-only latent.")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:  # unbatched [C,2,T]
        audio = audio.unsqueeze(0)
    if audio.ndim != 4:
        raise ValueError("h3_motion_context: expected audio latent [B,C,2,T], "
                         "got shape %s" % (tuple(audio.shape),))
    total_t = int(audio.shape[-1])
    frames = _pixel_frames(int(video.shape[2]))
    overhang = total_t - FRAME_RESCALE * frames
    if not (0.0 <= overhang < 1.0):
        _LOG.warning(
            "h3_motion_context: context_latent audio grid is unexpected "
            "(%d steps for %d frames); assuming no overhang.", total_t, frames)
        overhang = 0.0
    rt = int(round(a_frames / float(FPS) * AUDIO_HZ))
    if rt > total_t:
        _LOG.warning("h3_motion_context: asked for %d audio steps, the latent "
                     "has %d. Pinning all of it.", rt, total_t)
        rt = total_t
    if rt < 1:
        raise ValueError("h3_motion_context: audio window is empty")
    tail = audio[:1, ..., total_t - rt:].clone()
    return tail, rt, float(overhang)


class MiniMaxH3MotionContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "context_frames": ("IMAGE",),
                "context_length": ("INT", {
                    "default": 5, "min": 1, "max": 39,
                    "tooltip": "Frames of the previous clip to carry over. In "
                               "video mode only 1, 5, 22 and 39 are distinct; "
                               "anything else is snapped DOWN to the nearest so "
                               "the pinned run always ends at the clip's last "
                               "frame."}),
                "encode_mode": (["video", "frames"], {
                    "default": "video",
                    "tooltip": "video: one VAE call, motion lives inside the "
                               "latent, fewer rows. frames: one call per frame, "
                               "each pinned as a separate still."}),
                "anchor_mode": (["head", "before"], {
                    "default": "head",
                    "tooltip": "head: pinned frames occupy the first indices and "
                               "come back in the output, so trim them. before: "
                               "negative indices, nothing wasted, but the "
                               "coordinates overlap the text rows."}),
                "crop": (["disabled", "center"], {"default": "disabled"}),
                "audio_context_length": ("INT", {
                    "default": 22, "min": 0, "max": 240,
                    "tooltip": "Frames of tail audio to pin, independent of the "
                               "video window. 0 follows context_length. In "
                               "timeline mode the window is END-aligned with "
                               "the pinned video, so 22 with a 22-frame video "
                               "window overlays it exactly; longer windows "
                               "extend backwards into vacated coordinate "
                               "space (untested)."}),
                "audio_mode": (["timeline", "ref"], {
                    "default": "timeline",
                    "tooltip": "timeline: pinned audio gets coordinates on "
                               "this clip's own timeline, end-aligned with "
                               "the pinned video, so the model reads it as "
                               "this clip's sound so far and continues it. "
                               "ref: stock placement in a span before the "
                               "clip, which the model imitates (similar "
                               "music, not phase-locked) rather than "
                               "continues."}),
            },
            "optional": {
                "context_latent": ("LATENT", {
                    "tooltip": "Previous clip's SAMPLER OUTPUT latent (the same "
                               "one you wire into the decode nodes). When "
                               "supplied, the pinned audio is sliced straight "
                               "from it, skipping the decode/re-encode round "
                               "trip that dulls sound a little more at every "
                               "link of a chain. Takes priority over "
                               "context_audio; audio_vae is not needed on "
                               "this path."}),
                "audio_vae": ("VAE", {
                    "tooltip": "H3 audio VAE. Supply with context_audio to carry "
                               "the previous clip's tail sound across the join. "
                               "Not needed when context_latent is wired."}),
                "context_audio": ("AUDIO", {
                    "tooltip": "Audio of the previous clip. The tail matching the "
                               "pinned frames is encoded and pinned alongside "
                               "them. Ignored when context_latent is wired."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("conditioning", "trim_frames")
    FUNCTION = "apply"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Pin a run of consecutive frames from a previous clip as "
                   "never-denoised conditioning rows, so the model reads real "
                   "motion instead of guessing it from a single still.")

    def apply(self, conditioning, vae, latent, context_frames, context_length,
              encode_mode, anchor_mode, crop, audio_context_length=22,
              audio_mode="timeline", context_latent=None, audio_vae=None,
              context_audio=None):
        _ensure_h3_runtime_patches()

        video = _video_from_latent(latent)
        latent_t = int(video.shape[2])
        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16
        frame_count = _pixel_frames(latent_t)

        available = int(context_frames.shape[0])
        n = min(int(context_length), available)
        if n < 1:
            raise ValueError("h3_motion_context: context_frames is empty")
        if n < context_length:
            _LOG.warning("h3_motion_context: only %d frames supplied, pinning %d",
                         available, n)

        if encode_mode == "video":
            # snap down to the VAE grid BEFORE slicing, so the frames encoded
            # are exactly the frames the latent steps will cover (see
            # VIDEO_RUN_GRID). Slicing the last n and letting the VAE keep the
            # first `covered` of them would pin a run ending before the clip
            # does, and the join would jump by the difference.
            run = next(g for g in VIDEO_RUN_GRID if g <= n)
            if run != n:
                _LOG.warning(
                    "h3_motion_context: %d frames is off the VAE grid; pinning "
                    "the last %d instead (usable runs: 1, 5, 22, 39)", n, run)
            n = run

        if n >= frame_count:
            raise ValueError(
                "h3_motion_context: asked to pin %d frames into a %d frame clip. "
                "The pinned run must be a small fraction of the timeline."
                % (n, frame_count))

        # the LAST n frames of the incoming clip become the pinned run
        tail = _resize(context_frames[available - n:], width, height, crop)

        if encode_mode == "video":
            # one call; the VAE reads the batch axis as time and compresses
            enc = vae.encode(tail)
            if getattr(enc, "ndim", 0) != 5:
                raise ValueError(
                    "h3_motion_context: video-mode encode returned shape %s, "
                    "expected [B,C,T,H,W]. Try encode_mode=frames."
                    % (tuple(getattr(enc, "shape", ())),))
            steps = int(enc.shape[2])
            offsets = _step_offsets(steps)
            covered = _pixel_frames(steps)
            if covered != n:
                # n was snapped to the grid above, so a mismatch here means
                # the VAE's downscale formula changed underneath us and the
                # pinned content no longer lines up with the positions we
                # would write. Refuse rather than render a shifted join.
                raise RuntimeError(
                    "h3_motion_context: %d frames encoded to %d latent steps "
                    "covering %d frames; the VAE grid no longer matches "
                    "VIDEO_RUN_GRID. Upstream VAE change, refusing to run."
                    % (n, steps, covered))
            blocks = [enc[:, :, k:k + 1] for k in range(steps)]
            span = covered
        else:
            blocks, offsets = [], []
            for i in range(n):
                blocks.append(vae.encode(tail[i:i + 1]))
                offsets.append(i)
            span = n

        if anchor_mode == "before":
            indices = [o - span for o in offsets]
        else:
            indices = list(offsets)

        keyframes = []
        for p, blk in zip(indices, blocks):
            keyframes.append({
                # stock code accepts only 0 or frame_count-1 here; the real
                # position rides under MC_KEY and the layout patch applies it
                "resolved_frame_index": 0,
                MC_KEY: p,
                "latent": blk,
            })

        values = {
            "minimax_keyframes": keyframes,
            "minimax_frame_count": frame_count,
        }

        ref_audio_t = 0
        a_frames = 0
        audio_src = "off"
        if context_latent is not None or context_audio is not None:
            # the audio window is independent of the video one: audio cond
            # rows cost rows but never cost delivered frames
            a_frames = int(audio_context_length) or span
            if context_latent is not None:
                if context_audio is not None:
                    _LOG.info("h3_motion_context: both context_latent and "
                              "context_audio wired; using the latent (skips "
                              "one VAE round trip).")
                audio_latent, ref_audio_t, overhang = _audio_tail_from_latent(
                    context_latent, a_frames)
                audio_src = "latent"
            else:
                if audio_vae is None:
                    raise ValueError(
                        "h3_motion_context: context_audio supplied without "
                        "audio_vae. Wire the H3 audio VAE, or wire "
                        "context_latent instead.")
                audio_latent, ref_audio_t = _encode_audio_window(
                    audio_vae, context_audio, a_frames / float(FPS), tail=True)
                overhang = 0.0  # decoded audio was match_tail-cut at the frame
                audio_src = "vae"
            ref = {
                "kind": "audio",
                "ref_audio_t": ref_audio_t,
                "audio_latent": audio_latent,
            }
            if audio_mode == "timeline":
                # end-align the audio window with the pinned video: both are
                # the tail of clip A, so both must end at the same instant
                # of the new timeline -- frame `span` in head mode (where
                # A's last frame sits), frame 0 in before mode. On the
                # latent path the sliced content reaches `overhang` of a
                # step past A's last frame (H3 rounds its audio grid up),
                # so the end coordinate moves by exactly that much; the
                # layout patch takes a fractional frame index.
                end_frame = float(span if anchor_mode == "head" else 0)
                end_frame += overhang / FRAME_RESCALE
                ref[MC_AUDIO_KEY] = end_frame
        out = node_helpers.conditioning_set_values(conditioning, values)
        if ref_audio_t:
            # append=True preserves existing Ref2VA refs and places the
            # Motion Context timeline-audio ref last.
            out = node_helpers.conditioning_set_values(
                out, {"minimax_refs": [ref]}, append=True)

        trim = span if anchor_mode == "head" else 0
        _LOG.info("h3_motion_context: %s/%s, %d frames -> %d cond blocks at "
                  "indices %d..%d, %d frame clip at %dx%d, trim %d, audio %s",
                  encode_mode, anchor_mode, n, len(blocks),
                  indices[0], indices[-1], frame_count, width, height, trim,
                  ("%d frames -> %d latent steps (%.3fs) from %s, %s"
                   % (a_frames, ref_audio_t, ref_audio_t / AUDIO_HZ, audio_src,
                      "on the timeline ending at frame %.3f"
                      % float(ref.get(MC_AUDIO_KEY))
                      if audio_mode == "timeline" else "stock ref placement"))
                  if ref_audio_t else "off")
        return (out, trim)


class MiniMaxH3MotionContextTrim:
    """Drop the pinned head off a decoded clip, picture and sound together.

    The pinned frames occupy the start of the delivered timeline, so they
    have to come off before concatenating. Trimming only the images would
    leave the audio a full trim_frames longer than the video, and muxing
    those puts the whole soundtrack ahead of the picture by trim_frames/24
    seconds. At 5 frames that is 208ms, silent on ambience but squarely
    offbeat on anything with a pulse.

    So this takes both streams and removes the same span from each: whole
    frames from the images, the matching number of samples from the
    waveform. Wire trim_frames from the motion context node so the count
    follows whatever the encoder actually produced.

    The tail needs the same treatment for a different reason. H3's audio
    latent runs at 40 Hz against 24 fps picture, and FRAME_RESCALE is 5/3,
    so a 124 frame clip wants 206.67 audio steps and the layout rounds up
    to 207. Every clip therefore ships about 8.3 ms more sound than
    picture. Concatenate two and the second seam is out by 16.7 ms, three
    and it is 25 ms, and the error grows without bound down a chain. It
    reads as a faint dampening at the first join and a short click at
    later ones. Truncating the tail to exactly frames/fps stops it
    accumulating.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "trim_frames": ("INT", {"default": 0, "min": 0, "max": 4096}),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Decoded audio for the same clip. Trimmed by the "
                               "matching duration so sound stays locked to "
                               "picture. Leave unwired for silent clips."}),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                    "tooltip": "Frame rate used to convert the trim into an "
                               "audio duration. Must match what you feed "
                               "Create Video."}),
                "match_tail": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Truncate the audio so its duration equals "
                               "frames/fps exactly. H3 rounds its audio grid up, "
                               "so each clip carries about 8ms of extra sound "
                               "that accumulates at every join in a chain."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "trim"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Remove the leading pinned frames from a decoded H3 clip, "
                   "trimming picture and sound by the same duration.")

    def trim(self, images, trim_frames, audio=None, fps=24.0, match_tail=True):
        n = max(0, int(trim_frames))
        total = int(images.shape[0])
        if n >= total:
            raise ValueError(
                "h3_motion_context: asked to trim %d frames from a %d frame clip"
                % (n, total))
        out_images = images[n:] if n else images

        out_audio = audio
        if audio is not None:
            waveform = audio["waveform"]
            sr = int(audio["sample_rate"])
            seconds = n / float(fps)
            cut = int(round(seconds * sr))
            length = int(waveform.shape[-1])
            if cut >= length:
                raise ValueError(
                    "h3_motion_context: trimming %.3fs from %.3fs of audio would "
                    "leave nothing. Check that fps matches the clip."
                    % (seconds, length / sr))
            waveform = waveform[..., cut:]

            if match_tail:
                frames_left = total - n
                want = int(round(frames_left / float(fps) * sr))
                have = int(waveform.shape[-1])
                if have > want:
                    over = have - want
                    waveform = waveform[..., :want]
                    _LOG.info("h3_motion_context: tail trimmed %d samples "
                              "(%.2fms) so audio matches %d frames exactly",
                              over, over / sr * 1000.0, frames_left)
                elif have < want:
                    _LOG.warning("h3_motion_context: audio is %.2fms shorter than "
                                 "%d frames; leaving the tail alone",
                                 (want - have) / sr * 1000.0, frames_left)

            out_audio = {"waveform": waveform, "sample_rate": sr}
            _LOG.info("h3_motion_context: %d frames / %.4fs picture, %.4fs sound, "
                      "drift %.2fms",
                      total - n, (total - n) / float(fps),
                      int(waveform.shape[-1]) / sr,
                      abs((total - n) / float(fps) - int(waveform.shape[-1]) / sr) * 1000.0)
        elif n:
            _LOG.info("h3_motion_context: trimmed %d leading frames, %d remain. "
                      "No audio wired; if this clip has sound, mux it through "
                      "this node or it will run %.3fs ahead of the picture.",
                      n, total - n, n / float(fps))

        return (out_images, out_audio)


def _resolve_latent_path(path, clip_index=0):
    """Turn the loader's path input into a concrete file.

    Accepts an absolute path, a path relative to ComfyUI's output folder,
    or a directory (in either form). For a directory:

      clip_index == 0   the NEWEST .safetensors inside is used. Simple,
                        but NOT retry-safe: re-rolling a clip loads the
                        rejected attempt's own save (see the node docs).
                        Its run counter also numbers ATTEMPTS, not clips.
      clip_index  > 0   exactly that clip's slot is loaded: clip 1 is
                        *_00001.safetensors. Auto-mode files carry a
                        trailing underscore (*_00001_.safetensors) and
                        are never matched, because their numbers count
                        runs and could hold a reject.
    """
    p = (path or "").strip().strip('"').strip("'")
    if not p:
        p = "h3_context"
    candidates = [p, os.path.join(folder_paths.get_output_directory(), p)]
    for c in candidates:
        if os.path.isfile(c):
            return c
        if os.path.isdir(c):
            idx = int(clip_index)
            if idx > 0:
                # indexed slots use the natural name: clip 2 lives in
                # *_00002.safetensors. Auto-mode files carry a trailing
                # underscore (*_00002_.safetensors) and are deliberately
                # NOT matched: their numbers count runs, not clips, so a
                # reject could be sitting in any of them.
                endings = ("_%05d.safetensors" % idx,)
                files = [os.path.join(c, f) for f in os.listdir(c)
                         if f.endswith(endings)]
                if not files:
                    near = [f for f in os.listdir(c)
                            if f.endswith("_%05d_.safetensors" % idx)]
                    hint = ""
                    if near:
                        hint = (" Found %s, which is an auto-numbered save "
                                "(trailing underscore = numbered by RUN, so "
                                "it may be a reject). If it really is clip "
                                "%d, rename it to drop the trailing "
                                "underscore: %s" %
                                (near[0], idx,
                                 near[0].replace("_%05d_" % idx,
                                                 "_%05d" % idx)))
                    raise FileNotFoundError(
                        "h3_motion_context: no saved latent for clip %d "
                        "(no *_%05d.safetensors in %s).%s"
                        % (idx, idx, c, hint))
            else:
                files = [os.path.join(c, f) for f in os.listdir(c)
                         if f.endswith(".safetensors")]
                if not files:
                    raise FileNotFoundError(
                        "h3_motion_context: no saved latents in %s. Run a "
                        "clip with the Save Latent node first." % c)
            return max(files, key=os.path.getmtime)
    raise FileNotFoundError(
        "h3_motion_context: %r is neither a file nor a folder (also tried "
        "relative to the ComfyUI output directory)." % p)


class MiniMaxH3MotionContextSaveLatent:
    """Save an H3 AV latent to disk so the NEXT run can load it.

    Wiring the sampler's output straight into context_latent is a cycle:
    the sampler would be consuming its own result. The latent that motion
    context needs is the PREVIOUS clip's, which lives in the previous run
    -- so it has to cross runs through disk, the same way the frames and
    audio already do. Stock Save/Load Latent can't serialise H3's nested
    video/audio pair; this saves the two streams side by side.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "The sampler's output latent (the same one "
                               "you wire into the decode nodes)."}),
                "filename_prefix": ("STRING", {
                    "default": "h3_context/clip",
                    "tooltip": "Saved under the ComfyUI output folder. The "
                               "default keeps all chain latents in one "
                               "folder so the Load node can always pick "
                               "the newest."}),
                "clip_index": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "Which clip of the chain THIS is. Saves to "
                               "that clip's fixed slot, so a re-roll "
                               "overwrites its own reject instead of "
                               "stacking new files. Generating clip 2: "
                               "set 2 here and 1 on the Load node. 0 = "
                               "old behaviour, a new numbered file every "
                               "run (numbers count runs, not clips)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("latent_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Save the sampler's AV latent so the next run's Motion "
                   "Context node can pin audio from it via the matching "
                   "Load node.")

    def save(self, latent, filename_prefix, clip_index=0):
        parts = _streams_from_latent(latent)
        if len(parts) < 2:
            raise ValueError(
                "h3_motion_context: latent has no audio stream; wire the "
                "sampler output of an H3 AV graph.")
        video = parts[0].cpu().contiguous()
        audio = parts[1].cpu().contiguous()
        folder, filename, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory())
        if int(clip_index) > 0:
            # fixed slot with the natural name: clip 2 -> *_00002. A
            # re-roll of this clip overwrites its own save, so rejects
            # never accumulate or get loaded later. Auto mode (below)
            # keeps a trailing underscore, which is what excludes its
            # run-numbered files from indexed loading.
            path = os.path.join(folder, "%s_%05d.safetensors"
                                % (filename, int(clip_index)))
        else:
            path = os.path.join(folder, "%s_%05d_.safetensors"
                                % (filename, counter))
        _st_save({"video": video, "audio": audio}, path,
                 metadata={"format": "h3_motion_context_av_v1"})
        _LOG.info("h3_motion_context: saved AV latent to %s (video %s, "
                  "audio %s)", path, tuple(video.shape), tuple(audio.shape))
        return (path,)


class MiniMaxH3MotionContextLoadLatent:
    """Load a saved H3 AV latent for the context_latent input.

    clip_index means exactly what it says: set it to the clip you want to
    CONTINUE FROM, and that clip's slot is loaded. Generating clip 2 from
    clip 1: Load node 1, Save node 2. Re-rolling clip 2 changes nothing --
    it reloads slot 1 and overwrites slot 2's reject. Accept, then bump
    both numbers.

    At 0 it loads the newest file in the folder instead. Simple, but NOT
    retry-safe: a re-roll's newest file is the rejected attempt's own
    save, so the retry gets conditioned on the audio you just rejected.

    The output is ONLY for the Motion Context node's context_latent input.
    It is not a decodable latent -- do not wire it into VAE decode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_path": ("STRING", {
                    "default": "h3_context",
                    "tooltip": "A saved latent file, or a folder (relative "
                               "paths resolve against the ComfyUI output "
                               "directory). Pointing at a specific FILE "
                               "always loads that file, ignoring "
                               "clip_index."}),
                "clip_index": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "The clip to CONTINUE FROM: that clip's "
                               "slot is loaded. Generating clip 2 from "
                               "clip 1: set 1 here and 2 on the Save "
                               "node. 0 = newest file in the folder "
                               "(NOT retry-safe: a re-roll loads its own "
                               "rejected audio)."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Load a latent saved by H3 Motion Context Save Latent, "
                   "for the context_latent input only.")

    @classmethod
    def IS_CHANGED(cls, latent_path, clip_index=0):
        # the path string stays constant while the file behind it changes
        # (newest save, or an overwritten slot), so cache on the resolved
        # file identity instead -- otherwise ComfyUI would happily serve
        # a stale latent forever
        try:
            p = _resolve_latent_path(latent_path, clip_index)
            return "%s:%d" % (p, os.stat(p).st_mtime_ns)
        except Exception:
            return float("NaN")  # unresolvable: never cache

    def load(self, latent_path, clip_index=0):
        path = _resolve_latent_path(latent_path, clip_index)
        data = _st_load(path)
        if "video" not in data or "audio" not in data:
            raise ValueError(
                "h3_motion_context: %s is not an h3_motion_context latent "
                "(missing video/audio streams). Was it saved by the stock "
                "Save Latent node instead?" % path)
        _LOG.info("h3_motion_context: loaded AV latent from %s", path)
        # a plain list, not a NestedTensor: only this repo's context_latent
        # input accepts it, which is the point -- it cannot be mistaken
        # for a decodable latent without failing loudly downstream
        return ({"samples": [data["video"], data["audio"]]},)


class _DynamicInputs(dict):
    """Dynamic backend input map: accepts any key under any declared prefix.
    Fixed entries (keyword arguments) are stored as real dict items so they
    survive API serialization and enumerate normally.

    Legacy two-argument form (one prefix, one type) still works.
    """

    def __init__(self, *pairs, **fixed):
        self._prefixes = {}
        if len(pairs) == 2 and isinstance(pairs[0], str) \
                and isinstance(pairs[1], (tuple, list)):
            pairs = [pairs]
        # longest prefix wins, so "video_audio_" beats "video_"
        for prefix, types in sorted(pairs, key=lambda p: -len(p[0])):
            self._prefixes[prefix] = types
        for name, spec in fixed.items():
            dict.__setitem__(self, name, spec)

    def __contains__(self, key):
        return (isinstance(key, str)
                and any(key.startswith(p) for p in self._prefixes)) \
                or dict.__contains__(self, key)

    def __getitem__(self, key):
        if isinstance(key, str):
            for prefix, types in self._prefixes.items():
                if key.startswith(prefix):
                    return types
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

class MiniMaxH3CustomKeyframes:
    """Attach still-image H3 keyframes at arbitrary timeline positions.

    Each slot has a strength ("keyframe N strength") between 0.05 and 1.0
    that sets how much of the run the image stays pinned: the rows are
    pinned EXACT (clean, under the canonical 0.999 claim) while the video
    schedule's progress stays below the strength, then the block's tokens
    are dropped from the layout and the model's own stream covers the
    region with no reference at all. So 1.0 pins exactly, 0.9 almost the
    image, 0.5 pinned half then free re-render, 0.1 a light early-structure
    hint. Nothing noisy is ever shown to the model.
    """

    MAX_KEYFRAMES = 32

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": (
                    "CONDITIONING",
                    {
                        "tooltip": (
                            "H3 conditioning. The node replaces its complete "
                            "minimax_keyframes list with the keyframes below."
                        )
                    },
                ),
                "vae": (
                    "VAE",
                    {
                        "tooltip": (
                            "MiniMax H3 video VAE used to encode each still."
                        )
                    },
                ),
                "latent": (
                    "LATENT",
                    {
                        "tooltip": (
                            "Target MiniMax H3 AV latent; defines resolution "
                            "and exact frame count."
                        )
                    },
                ),
                "keyframe_state": (
                    "STRING",
                    {
                        "default": (
                            '{"count":3,"positions":[1,22,79],'
                            '"strengths":[1,1,1]}'
                        ),
                        "multiline": False,
                        "tooltip": (
                            "Internal UI state. Normally managed by the "
                            "keyframe position and strength controls."
                        ),
                    },
                ),
                "indexing": (
                    ["1-based", "0-based"],
                    {"default": "1-based"},
                ),
                "crop": (
                    ["disabled", "center"],
                    {"default": "disabled"},
                ),
            },
            "optional": _DynamicInputs("keyframe_image_", ("IMAGE",)),
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Pin still-image MiniMax H3 keyframes at arbitrary output-frame "
        "positions. Starts with 3 keyframes; use + Add keyframe to add more. "
        "Strength 1.0 pins the image exactly (default); lower values let "
        "the model vary the content more."
    )

    def apply(
        self,
        conditioning,
        vae,
        latent,
        keyframe_state,
        indexing="1-based",
        crop="disabled",
        **kwargs,
    ):
        _ensure_h3_runtime_patches()

        try:
            state = json.loads(keyframe_state or "{}")
        except Exception as exc:
            raise ValueError(
                "h3_motion_context: invalid H3 Custom Keyframes UI state"
            ) from exc

        positions = state.get("positions", [])
        count = int(state.get("count", len(positions)))
        strengths = state.get("strengths", [])

        if count < 1 or count > self.MAX_KEYFRAMES:
            raise ValueError(
                "h3_motion_context: Custom Keyframes count must be 1..%d"
                % self.MAX_KEYFRAMES
            )
        if len(positions) < count:
            raise ValueError(
                "h3_motion_context: %d keyframe slots but only %d saved "
                "positions" % (count, len(positions))
            )
        if len(strengths) < count:
            strengths = [1.0] * count
        strengths = [min(1.0, max(0.05, float(s)))
                     for s in strengths[:count]]

        video = _video_from_latent(latent)
        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16
        frame_count = _pixel_frames(int(video.shape[2]))

        anchors = []
        for slot in range(1, count + 1):
            raw_position = int(positions[slot - 1])
            pixel_index = (
                raw_position - 1
                if indexing == "1-based"
                else raw_position
            )

            if pixel_index < 0 or pixel_index >= frame_count:
                low, high = (
                    (1, frame_count)
                    if indexing == "1-based"
                    else (0, frame_count - 1)
                )
                raise ValueError(
                    "h3_motion_context: keyframe %d position %d is "
                    "outside %d..%d"
                    % (slot, raw_position, low, high)
                )

            image = kwargs.get("keyframe_image_%d" % slot)
            if image is None:
                raise ValueError(
                    "h3_motion_context: keyframe %d has no image connected"
                    % slot
                )
            if getattr(image, "ndim", 0) != 4:
                raise ValueError(
                    "h3_motion_context: keyframe %d expected IMAGE "
                    "[B,H,W,C]" % slot
                )
            if int(image.shape[0]) != 1:
                raise ValueError(
                    "h3_motion_context: keyframe %d must receive exactly "
                    "one image, not a batch of %d"
                    % (slot, int(image.shape[0]))
                )

            anchors.append((pixel_index, slot, image))

        anchors.sort(key=lambda item: item[0])

        for i in range(1, len(anchors)):
            if anchors[i - 1][0] == anchors[i][0]:
                displayed = (
                    anchors[i][0] + 1
                    if indexing == "1-based"
                    else anchors[i][0]
                )
                raise ValueError(
                    "h3_motion_context: duplicate keyframe position %d"
                    % displayed
                )

        keyframes = []
        for pixel_index, slot, image in anchors:
            resized = _resize(image, width, height, crop)
            encoded = vae.encode(resized)

            if (
                getattr(encoded, "ndim", 0) != 5
                or int(encoded.shape[2]) != 1
            ):
                raise ValueError(
                    "h3_motion_context: keyframe %d encoded to %s; "
                    "expected one H3 still latent [B,C,1,H,W]"
                    % (
                        slot,
                        tuple(getattr(encoded, "shape", ())),
                    )
                )

            keyframes.append({
                # Stock PackedLayout accepts frame 0 here. The real temporal
                # location is applied lazily through MC_KEY.
                "resolved_frame_index": 0,
                MC_KEY: int(pixel_index),
                MC_VIDEO_STRENGTH: strengths[slot - 1],
                "latent": encoded,
            })

        out = node_helpers.conditioning_set_values(
            conditioning,
            {
                "minimax_keyframes": keyframes,
                "minimax_frame_count": frame_count,
            },
        )

        shown = [
            p + 1 if indexing == "1-based" else p
            for p, _, _ in anchors
        ]
        _LOG.info(
            "h3_motion_context: Custom Keyframes pinned %d anchors at %s "
            "in a %d-frame %dx%d target",
            len(keyframes),
            shown,
            frame_count,
            width,
            height,
        )

        return (out,)


class MiniMaxH3CustomAudio:
    """Pin audio clips at arbitrary positions of the H3 output timeline.

    Stock audio refs sit in a span before the clip and are merely imitated.
    Every block attached here is instead MOVED onto the target clip's own
    audio timeline, with its sound (or its start, in start mode) landing at
    the requested frame, exactly how custom keyframes anchor the video. The
    injected sound is never denoised, so the model hears it as established
    fact and must generate the clip's audio around it -- before it, after
    it, or both.

    align=end:  the audio block ENDS at the chosen frame. Sound before that
                instant is pinned, the model continues from it.
    align=start: the audio block STARTS at the chosen frame. Sound from that
                instant on is pinned, the model leads into it.

    Each slot has a strength ("audio N strength") between 0.05 and 1.0 that
    sets how much of the clip stays pinned: the rows are pinned EXACT (clean,
    no noise) while the audio schedule's progress stays below the strength,
    then the block's tokens are dropped from the layout and the model's own
    stream covers the region with no reference at all. So 1.0 pins exactly,
    0.9 almost the clip, 0.5 pinned half then free re-render, 0.1 a light
    early-structure hint. Nothing noisy is ever shown to the model.
    """

    MAX_AUDIOS = 16

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": (
                    "CONDITIONING",
                    {
                        "tooltip": (
                            "H3 conditioning. Custom audio blocks are appended "
                            "to any existing minimax_refs (Ref2VA refs, H3 "
                            "Motion Context audio)."
                        )
                    },
                ),
                "audio_vae": (
                    "VAE",
                    {
                        "tooltip": (
                            "MiniMax H3 audio VAE used to encode each audio "
                            "clip."
                        )
                    },
                ),
                "latent": (
                    "LATENT",
                    {
                        "tooltip": (
                            "Target MiniMax H3 AV latent; defines the frame "
                            "count the audio positions are measured against."
                        )
                    },
                ),
                "audio_state": (
                    "STRING",
                    {
                        "default": '{"count":1,"positions":[1],"strengths":[1]}',
                        "multiline": False,
                        "tooltip": (
                            "Internal UI state. Normally managed by the "
                            "audio position and strength controls."
                        ),
                    },
                ),
                "indexing": (
                    ["1-based", "0-based"],
                    {"default": "1-based"},
                ),
                "align": (
                    ["end", "start"],
                    {
                        "default": "end",
                        "tooltip": (
                            "end: the audio block finishes at the chosen "
                            "frame (sound already here, the model continues "
                            "it). start: the audio block begins at the "
                            "chosen frame (sound arrives there, the model "
                            "leads into it)."
                        ),
                    },
                ),
            },
            "optional": _DynamicInputs("audio_", ("AUDIO",)),
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Pin MiniMax H3 audio clips at arbitrary output-frame positions, "
        "end- or start-aligned. Starts with 1 audio slot; use + Add audio "
        "for more. Each clip has its own strength: 1.0 pins it exactly "
        "(default), lower values let the model vary the sound more."
    )

    def apply(
        self,
        conditioning,
        audio_vae,
        latent,
        audio_state,
        indexing="1-based",
        align="end",
        **kwargs,
    ):
        _ensure_h3_runtime_patches()

        try:
            state = json.loads(audio_state or "{}")
        except Exception as exc:
            raise ValueError(
                "h3_motion_context: invalid H3 Custom Audio UI state"
            ) from exc

        positions = state.get("positions", [])
        count = int(state.get("count", len(positions)))
        strengths = state.get("strengths", [])

        if count < 1 or count > self.MAX_AUDIOS:
            raise ValueError(
                "h3_motion_context: Custom Audio count must be 1..%d"
                % self.MAX_AUDIOS
            )
        if len(positions) < count:
            raise ValueError(
                "h3_motion_context: %d audio slots but only %d saved "
                "positions" % (count, len(positions))
            )
        if len(strengths) < count:
            strengths = [1.0] * count
        strengths = [
            min(1.0, max(0.05, float(s)))
            for s in strengths[:count]
        ]

        frame_count = _pixel_frames(int(_video_from_latent(latent).shape[2]))

        anchors = []
        for slot in range(1, count + 1):
            raw_position = int(positions[slot - 1])
            zero_based = (
                raw_position - 1 if indexing == "1-based" else raw_position
            )

            if align == "end":
                # the block ends at the END of 1-based frame `position`
                # (sound covers frames 1..position). position is the 1-based
                # raw value; a 0-based slot maps to the same end instant.
                if zero_based < 0 or zero_based >= frame_count:
                    low, high = (
                        (1, frame_count)
                        if indexing == "1-based"
                        else (0, frame_count - 1)
                    )
                    raise ValueError(
                        "h3_motion_context: audio %d end position %d is "
                        "outside %d..%d" % (slot, raw_position, low, high)
                    )
                window_frames = zero_based + 1
            else:
                if zero_based < 0 or zero_based >= frame_count:
                    low, high = (
                        (1, frame_count)
                        if indexing == "1-based"
                        else (0, frame_count - 1)
                    )
                    raise ValueError(
                        "h3_motion_context: audio %d start position %d is "
                        "outside %d..%d" % (slot, raw_position, low, high)
                    )
                window_frames = frame_count - zero_based

            audio = kwargs.get("audio_%d" % slot)
            if audio is None:
                raise ValueError(
                    "h3_motion_context: audio %d has no clip connected" % slot
                )
            waveform = audio.get("waveform")
            if getattr(waveform, "ndim", 0) != 3 or int(waveform.shape[-1]) < 1:
                raise ValueError(
                    "h3_motion_context: audio %d expected an AUDIO clip "
                    "[B,C,L]" % slot
                )

            z, rt = _encode_audio_window(
                audio_vae, audio, window_frames / float(FPS),
                tail=(align == "end"))
            if rt < 1:
                raise ValueError(
                    "h3_motion_context: audio %d encoded to zero latent "
                    "steps" % slot
                )

            if align == "end":
                # sound ends at the end of 1-based frame `position`
                end_frame = float(zero_based + 1)
            else:
                # sound starts at 0-based frame `zero_based` and covers
                # rt 40 Hz steps = rt * 3/5 frame units of timeline
                end_frame = float(zero_based) + rt * 3.0 / 5.0

            anchors.append((end_frame, slot, z, rt, strengths[slot - 1]))

        anchors.sort(key=lambda item: item[0])

        for i in range(len(anchors)):
            for j in range(i + 1, len(anchors)):
                if abs(anchors[i][0] - anchors[j][0]) < 1e-6:
                    raise ValueError(
                        "h3_motion_context: audio %d and %d would both end at "
                        "frame %.3f. Pick different positions."
                        % (anchors[i][1], anchors[j][1], anchors[i][0])
                    )

        refs = [
            {
                "kind": "audio",
                "ref_audio_t": rt,
                "audio_latent": z,
                MC_AUDIO_KEY: end_frame,
                MC_AUDIO_STRENGTH: strength,
            }
            for end_frame, _, z, rt, strength in anchors
        ]

        out = node_helpers.conditioning_set_values(
            conditioning, {"minimax_refs": refs}, append=True)

        _LOG.info(
            "h3_motion_context: Custom Audio pinned %d blocks (%s-aligned) "
            "ending at %.3f..%.3f in a %d-frame clip",
            len(refs), align, anchors[0][0], anchors[-1][0], frame_count,
        )
        return (out,)


class MiniMaxH3CustomVideo:
    """Pin full video clips onto the H3 output timeline.

    Each clip is encoded in one VAE call; every latent step becomes a cond
    block anchored at its own pixel frame, so the motion lives inside the
    latents like a run of keyframes. An optional audio track per clip is
    windowed to that clip's duration (a longer track is cut from its start)
    and end-aligned with the clip's last frame.

    Each slot has a strength ("video N strength") between 0.05 and 1.0 that
    sets how much of the clip stays pinned: the clip is pinned EXACT (clean
    rows under the canonical 0.999 claim) for the first `strength`-fraction
    of the run, then its tokens are dropped from the layout and the model's
    own stream covers the region with no reference at all. So 1.0 pins
    exactly, 0.9 almost the clip, 0.5 pinned half then free re-render,
    0.1 a light early-structure hint. Nothing noisy is ever shown to the
    model. The clip's audio track follows the same strength on the same
    schedule (audio timeline, shifted via time_shift_sigma).
    """

    MAX_VIDEOS = 8

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "H3 conditioning. The node replaces its "
                               "minimax_keyframes list with the videos' cond "
                               "blocks and appends any audio refs to existing "
                               "refs."}),
                "vae": ("VAE", {
                    "tooltip": "MiniMax H3 video VAE used to encode the "
                               "clips."}),
                "latent": ("LATENT", {
                    "tooltip": "Target MiniMax H3 AV latent; defines "
                               "resolution and exact frame count."}),
                "video_state": ("STRING", {
                    "default": '{"count":1,"positions":[1],"strengths":[1]}',
                    "multiline": False,
                    "tooltip": "Internal UI state. Normally managed by the "
                               "video position and strength controls."}),
                "indexing": (["1-based", "0-based"], {"default": "1-based"}),
                "crop": (["disabled", "center"], {"default": "disabled"}),
            },
            "optional": _DynamicInputs(
                ("video_", ("IMAGE",)),
                ("video_audio_", ("AUDIO",)),
                audio_vae=("VAE", {
                    "tooltip": "H3 audio VAE. Required when any clip has an "
                               "audio track wired."}),
            ),
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Pin full video clips, each with its own position, "
                   "strength and optional audio track, onto the H3 timeline. "
                   "Starts with 1 video slot; use + Add video for more. "
                   "Strength 1.0 pins the clip exactly (default); lower "
                   "values let the model vary the content more.")

    def apply(self, conditioning, vae, latent, video_state,
              indexing="1-based", crop="disabled", audio_vae=None, **kwargs):
        _ensure_h3_runtime_patches()

        try:
            state = json.loads(video_state or "{}")
        except Exception as exc:
            raise ValueError(
                "h3_motion_context: invalid H3 Custom Video UI state"
            ) from exc

        positions = state.get("positions", [])
        count = int(state.get("count", len(positions)))
        strengths = state.get("strengths", [])

        if count < 1 or count > self.MAX_VIDEOS:
            raise ValueError(
                "h3_motion_context: Custom Video count must be 1..%d"
                % self.MAX_VIDEOS)
        if len(positions) < count:
            raise ValueError(
                "h3_motion_context: %d video slots but only %d saved "
                "positions" % (count, len(positions)))
        if len(strengths) < count:
            strengths = [1.0] * count
        strengths = [min(1.0, max(0.05, float(s)))
                     for s in strengths[:count]]

        target = _video_from_latent(latent)
        width = int(target.shape[4]) * 16
        height = int(target.shape[3]) * 16
        frame_count = _pixel_frames(int(target.shape[2]))

        keyframes = []
        audio_refs = []
        infos = []
        for slot in range(1, count + 1):
            raw_position = int(positions[slot - 1])
            zero_based = (raw_position - 1 if indexing == "1-based"
                          else raw_position)

            video = kwargs.get("video_%d" % slot)
            if video is None:
                raise ValueError(
                    "h3_motion_context: video %d has no clip connected" % slot)
            if getattr(video, "ndim", 0) != 4 or int(video.shape[0]) < 1:
                raise ValueError(
                    "h3_motion_context: video %d expected IMAGE frames "
                    "[B,H,W,C]" % slot)

            n = int(video.shape[0])
            run = next(g for g in VIDEO_RUN_GRID if g <= n)
            if run != n:
                _LOG.warning(
                    "h3_motion_context: video %d has %d frames, off the VAE "
                    "grid; pinning the first %d (usable runs: 1, 5, 22, 39)",
                    slot, n, run)

            if zero_based < 0 or zero_based + run > frame_count:
                raise ValueError(
                    "h3_motion_context: video %d does not fit: %d frames "
                    "starting at position %d in a %d frame clip"
                    % (slot, run, raw_position, frame_count))

            enc = vae.encode(_resize(video[:run], width, height, crop))
            steps = int(enc.shape[2])
            if _pixel_frames(steps) != run:
                raise RuntimeError(
                    "h3_motion_context: video %d encoded %d frames to %d "
                    "latent steps; the VAE grid no longer matches "
                    "VIDEO_RUN_GRID. Upstream VAE change, refusing to run."
                    % (slot, run, steps))

            strength = strengths[slot - 1]
            offsets = _step_offsets(steps)
            for k, off in enumerate(offsets):
                keyframes.append({
                    "resolved_frame_index": 0,
                    MC_KEY: zero_based + off,
                    MC_VIDEO_STRENGTH: strength,
                    "latent": enc[:, :, k:k + 1],
                })

            audio_info = "off"
            audio = kwargs.get("video_audio_%d" % slot)
            if audio is not None:
                if audio_vae is None:
                    raise ValueError(
                        "h3_motion_context: video %d has an audio track but "
                        "no audio_vae. Wire the H3 audio VAE." % slot)
                audio_latent, ref_audio_t = _encode_audio_window(
                    audio_vae, audio, run / float(FPS), tail=False)
                # end-aligned with this video's last frame: the block ends
                # at 1-based frame zero_based + run
                audio_refs.append({
                    "kind": "audio",
                    "ref_audio_t": ref_audio_t,
                    "audio_latent": audio_latent,
                    MC_AUDIO_KEY: float(zero_based + run),
                    MC_AUDIO_STRENGTH: strength,
                })
                audio_info = "%d latent steps" % ref_audio_t

            infos.append((slot, zero_based, zero_based + run - 1, run,
                          audio_info))

        out = node_helpers.conditioning_set_values(conditioning, {
            "minimax_keyframes": keyframes,
            "minimax_frame_count": frame_count,
        })
        if audio_refs:
            out = node_helpers.conditioning_set_values(
                out, {"minimax_refs": audio_refs}, append=True)

        _LOG.info(
            "h3_motion_context: Custom Video pinned %d clips (%d cond blocks "
            "total) in a %d-frame clip: %s",
            count, len(keyframes), frame_count,
            ", ".join("video %d = %d frames at %d..%d, audio %s"
                      % (slot, run, start, end, ai)
                      for slot, start, end, run, ai in infos),
        )
        return (out,)


class MiniMaxH3Timeline:
    """One node, one timeline: still images, video clips and audio clips.

    A single ordered list of clips, each pinned at a 1-based start frame:
      image  a still, pinned at its frame (an H3 custom keyframe)
      video  a full clip; every latent step is pinned at its own frame and
             its audio track (video_audio_N) rides the audio timeline
      audio  a window of sound pinned on the audio track

    A video's audio is linked by default: it follows the clip's position
    and length. Set audio_link false in the state to move and trim it
    independently (audio_start / audio_len / audio_align). Audio windows
    are cut from the head ("head") or the tail ("tail") of their source.

    Video and image placements are structural: they raise when they do not
    fit the target clip. Audio windows are contextual: out-of-range starts
    are parked at the last frame with a warning, never raised. Per-clip
    strength rides MC_VIDEO_STRENGTH / MC_AUDIO_STRENGTH exactly like the
    per-type custom nodes.
    """

    MAX_CLIPS = 32

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "H3 conditioning. The node replaces its "
                               "minimax_keyframes list with the timeline's "
                               "blocks and appends its audio refs."}),
                "video vae": ("VAE", {
                    "tooltip": "MiniMax H3 video VAE used to encode the "
                               "images and video clips."}),
                "audio vae": ("VAE", {
                    "tooltip": "H3 audio VAE. Required when any clip has "
                               "audio wired."}),
                "latent": ("LATENT", {
                    "tooltip": "Target MiniMax H3 AV latent; defines "
                               "resolution and exact frame count."}),
                "timeline_state": ("STRING", {
                    "default": '{"clips":[]}',
                    "multiline": False,
                    "tooltip": "Internal UI state. Normally managed by the "
                               "timeline widget."}),
                "crop": (["disabled", "center"], {"default": "disabled"}),
            },
            "optional": _DynamicInputs(
                ("video_", ("IMAGE",)),
                ("video_audio_", ("AUDIO",)),
                ("image_", ("IMAGE",)),
                ("audio_", ("AUDIO",)),
            ),
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Video-editor timeline: drop still images, video clips "
                   "and audio clips at any frame. A video's sound starts "
                   "linked to it; unlink to move or trim it alone.")

    @staticmethod
    def _audio_ref(audio_vae, audio, idx, strength, a_start, a_len, align):
        """Build one timeline audio ref ending at frame a_start + a_len."""
        if audio_vae is None:
            raise ValueError(
                "h3_motion_context: clip %d has audio but no audio_vae. "
                "Wire the H3 audio VAE." % idx)
        waveform = audio.get("waveform")
        if getattr(waveform, "ndim", 0) != 3 or int(waveform.shape[-1]) < 1:
            raise ValueError(
                "h3_motion_context: clip %d expected an AUDIO clip [B,C,L]"
                % idx)
        z, rt = _encode_audio_window(
            audio_vae, audio, a_len / float(FPS), tail=(align == "tail"))
        if rt < 1:
            raise ValueError(
                "h3_motion_context: clip %d audio encoded to zero latent "
                "steps" % idx)
        return {
            "kind": "audio",
            "ref_audio_t": rt,
            "audio_latent": z,
            MC_AUDIO_KEY: float(a_start - 1 + a_len),
            MC_AUDIO_STRENGTH: strength,
        }

    @staticmethod
    def _fit_audio(start, length, frame_count, idx):
        """Clamp an audio window into the clip. Audio rows are contextual,
        so an out-of-range start is parked at the last frame with a warning
        instead of killing the run."""
        start = max(1, int(start))
        length = max(1, int(length))
        end = start - 1 + length
        if start - 1 >= frame_count:
            _LOG.warning(
                "h3_motion_context: audio clip %d starts at frame %d, "
                "beyond the %d frame clip; parked at the last frame",
                idx, start, frame_count)
            return frame_count, 1
        if end > frame_count:
            _LOG.warning(
                "h3_motion_context: audio clip %d window cut from %d to %d "
                "frames to fit the %d frame clip",
                idx, length, frame_count - start + 1, frame_count)
            length = frame_count - start + 1
        return start, length

    def apply(self, conditioning, latent, timeline_state,
              crop="disabled", **kwargs):
        # input keys carry spaces ("video vae" / "audio vae"), so they only
        # arrive through **kwargs
        vae = kwargs.get("video vae")
        audio_vae = kwargs.get("audio vae")
        _ensure_h3_runtime_patches()

        try:
            state = json.loads(timeline_state or "{}")
        except Exception as exc:
            raise ValueError(
                "h3_motion_context: invalid H3 Timeline UI state") from exc

        clips = state.get("clips") or []
        if not clips:
            raise ValueError("h3_motion_context: Timeline has no clips")
        if len(clips) > self.MAX_CLIPS:
            raise ValueError(
                "h3_motion_context: Timeline clip count must be 1..%d"
                % self.MAX_CLIPS)

        target = _video_from_latent(latent)
        width = int(target.shape[4]) * 16
        height = int(target.shape[3]) * 16
        frame_count = _pixel_frames(int(target.shape[2]))

        keyframes, refs = [], []
        infos = []
        for idx, clip in enumerate(clips, 1):
            slot = int(clip.get("id") or idx)
            kind = clip.get("kind")
            if kind not in ("image", "video", "audio"):
                raise ValueError(
                    "h3_motion_context: timeline clip %d has unknown kind %r"
                    % (idx, kind))
            start = int(clip.get("start") or 1)
            if start < 1:
                raise ValueError(
                    "h3_motion_context: timeline clip %d start %d is below 1"
                    % (idx, start))
            strength = min(1.0, max(0.05, float(clip.get("strength") or 1.0)))
            zero = start - 1

            if kind == "video":
                frames = kwargs.get("video_%d" % slot)
                audio = kwargs.get("video_audio_%d" % slot)
                src_start = max(0, int(clip.get("src_start") or 0))
                fmedia = clip.get("file")
                if fmedia:
                    data = _load_media_file(fmedia)
                    frames = data["frames"]
                    if frames is None:
                        raise ValueError(
                            "h3_motion_context: video clip %d file %r has no "
                            "video stream"
                            % (idx, fmedia.get("name")))
                    frames = frames[src_start:]
                    if audio is None and clip.get("audio_link", True) \
                            and data["audio"] is not None:
                        audio = _slice_audio(
                            data["audio"], src_start / float(FPS))
                if frames is None:
                    raise ValueError(
                        "h3_motion_context: video clip %d has no frames "
                        "connected" % idx)
                if getattr(frames, "ndim", 0) != 4 or int(frames.shape[0]) < 1:
                    raise ValueError(
                        "h3_motion_context: video clip %d expected IMAGE "
                        "frames [B,H,W,C]" % idx)
                want = min(max(1, int(clip.get("len") or 22)),
                           int(frames.shape[0]))
                run = next(g for g in VIDEO_RUN_GRID if g <= want)
                if run != want:
                    _LOG.warning(
                        "h3_motion_context: video clip %d wants %d frames, "
                        "off the VAE grid; pinning the first %d (usable "
                        "runs: 1, 5, 22, 39)", idx, want, run)
                if zero + run > frame_count:
                    raise ValueError(
                        "h3_motion_context: video clip %d does not fit: %d "
                        "frames at frame %d in a %d frame clip"
                        % (idx, run, start, frame_count))
                enc = vae.encode(_resize(frames[:run], width, height, crop))
                steps = int(enc.shape[2])
                if _pixel_frames(steps) != run:
                    raise RuntimeError(
                        "h3_motion_context: video clip %d encoded %d frames "
                        "to %d latent steps; the VAE grid no longer matches "
                        "VIDEO_RUN_GRID. Upstream VAE change, refusing to run."
                        % (idx, run, steps))
                for k, off in enumerate(_step_offsets(steps)):
                    keyframes.append({
                        "resolved_frame_index": 0,
                        MC_KEY: zero + off,
                        MC_VIDEO_STRENGTH: strength,
                        "latent": enc[:, :, k:k + 1],
                    })

                audio = kwargs.get("video_audio_%d" % slot)
                if audio is not None:
                    if clip.get("audio_link", True):
                        a_start, a_len, align = start, run, "head"
                    else:
                        a_start, a_len = self._fit_audio(
                            clip.get("audio_start") or start,
                            clip.get("audio_len") or run,
                            frame_count, idx)
                        align = clip.get("audio_align", "head")
                    refs.append(self._audio_ref(
                        audio_vae, audio, idx, strength,
                        a_start, a_len, align))
                    link = "linked" if clip.get("audio_link", True) else \
                        "unlinked"
                    infos.append(
                        "clip %d: video %d..%d + audio %d..%d (%s)"
                        % (idx, zero + 1, zero + run, a_start,
                           a_start + a_len - 1, link))
                else:
                    infos.append(
                        "clip %d: video %d..%d, silent"
                        % (idx, zero + 1, zero + run))

            elif kind == "image":
                if zero >= frame_count:
                    raise ValueError(
                        "h3_motion_context: image clip %d at frame %d is "
                        "outside 1..%d" % (idx, start, frame_count))
                image = kwargs.get("image_%d" % slot)
                fmedia = clip.get("file")
                if fmedia:
                    image = _load_image_file(fmedia)
                if image is None:
                    raise ValueError(
                        "h3_motion_context: image clip %d has no image "
                        "connected" % idx)
                if int(image.shape[0]) != 1:
                    raise ValueError(
                        "h3_motion_context: image clip %d must receive "
                        "exactly one image, not a batch of %d"
                        % (idx, int(image.shape[0])))
                encoded = vae.encode(_resize(image, width, height, crop))
                if getattr(encoded, "ndim", 0) != 5 \
                        or int(encoded.shape[2]) != 1:
                    raise ValueError(
                        "h3_motion_context: image clip %d encoded to %s; "
                        "expected one H3 still latent [B,C,1,H,W]"
                        % (idx, tuple(getattr(encoded, "shape", ()))))
                keyframes.append({
                    "resolved_frame_index": 0,
                    MC_KEY: zero,
                    MC_VIDEO_STRENGTH: strength,
                    "latent": encoded,
                })
                infos.append("clip %d: image at frame %d" % (idx, start))

            else:  # audio
                if audio_vae is None:
                    raise ValueError(
                        "h3_motion_context: audio clip %d has no audio_vae. "
                        "Wire the H3 audio VAE." % idx)
                a_start, a_len = self._fit_audio(
                    start, clip.get("len") or 22, frame_count, idx)
                audio = kwargs.get("audio_%d" % slot)
                fmedia = clip.get("file")
                if fmedia:
                    data = _load_media_file(fmedia)
                    audio = data["audio"]
                    if audio is None:
                        raise ValueError(
                            "h3_motion_context: audio clip %d file %r has no "
                            "audio stream"
                            % (idx, fmedia.get("name")))
                    audio = _slice_audio(
                        audio,
                        max(0, int(clip.get("src_start") or 0)) / float(FPS))
                if audio is None:
                    raise ValueError(
                        "h3_motion_context: audio clip %d has no clip "
                        "connected" % idx)
                refs.append(self._audio_ref(
                    audio_vae, audio, idx, strength,
                    a_start, a_len, clip.get("align", "head")))
                infos.append("clip %d: audio %d..%d"
                             % (idx, a_start, a_start + a_len - 1))

        keyframes.sort(key=lambda kf: kf[MC_KEY])
        out = node_helpers.conditioning_set_values(conditioning, {
            "minimax_keyframes": keyframes,
            "minimax_frame_count": frame_count,
        })
        if refs:
            out = node_helpers.conditioning_set_values(
                out, {"minimax_refs": refs}, append=True)

        _LOG.info(
            "h3_motion_context: Timeline pinned %d cond blocks + %d audio "
            "refs in a %d-frame clip: %s",
            len(keyframes), len(refs), frame_count, "; ".join(infos))
        return (out,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3MotionContext": MiniMaxH3MotionContext,
    "MiniMaxH3MotionContextTrim": MiniMaxH3MotionContextTrim,
    "MiniMaxH3MotionContextSaveLatent": MiniMaxH3MotionContextSaveLatent,
    "MiniMaxH3MotionContextLoadLatent": MiniMaxH3MotionContextLoadLatent,
    "MiniMaxH3CustomKeyframes": MiniMaxH3CustomKeyframes,
    "MiniMaxH3CustomAudio": MiniMaxH3CustomAudio,
    "MiniMaxH3CustomVideo": MiniMaxH3CustomVideo,
    "MiniMaxH3Timeline": MiniMaxH3Timeline,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3MotionContext": "H3 Motion Context",
    "MiniMaxH3MotionContextTrim": "H3 Motion Context Trim",
    "MiniMaxH3MotionContextSaveLatent": "H3 Motion Context Save Latent",
    "MiniMaxH3MotionContextLoadLatent": "H3 Motion Context Load Latent",
    "MiniMaxH3CustomKeyframes": "H3 Custom Keyframes",
    "MiniMaxH3CustomAudio": "H3 Custom Audio",
    "MiniMaxH3CustomVideo": "H3 Custom Video",
    "MiniMaxH3Timeline": "H3 Timeline Editor",
}
