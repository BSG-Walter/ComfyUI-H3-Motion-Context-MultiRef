"""MiniMax H3 Timeline Editor.

Visual video-editor timeline node for MiniMax H3, using native ComfyUI core
minimax_keyframes guides.
"""

import json
import logging
import math
import os

import av
import numpy as np
import torch
import folder_paths
import node_helpers
import comfy.utils
from comfy_extras.nodes_audio import f32_pcm
from PIL import Image, ImageOps

try:
    import torchaudio
except ImportError:
    torchaudio = None

_LOG = logging.getLogger("h3_timeline")

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
FPS = 24


def _pixel_frames(latent_t):
    """Pixel frames covered by latent_t latent steps."""
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))


def _resize(image, width, height, crop="disabled"):
    """Resize image tensor [B, H, W, C] to target width/height."""
    if image is None:
        return None
    samples = image.movedim(-1, 1)  # [B, C, H, W]
    if crop == "center":
        h_ratio = height / float(samples.shape[2])
        w_ratio = width / float(samples.shape[3])
        scale = max(h_ratio, w_ratio)
        new_h = int(round(samples.shape[2] * scale))
        new_w = int(round(samples.shape[3] * scale))
        samples = comfy.utils.common_upscale(samples, new_w, new_h, "lanczos", crop="center")
        samples = samples[:, :, :height, :width]
    else:
        samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop="disabled")
    return samples.movedim(1, -1)


def _normalize_audio_waveform(wav):
    """Normalize any audio waveform tensor shape to [1, 2, L] stereo."""
    if wav is None:
        return None
    if not isinstance(wav, torch.Tensor):
        wav = torch.as_tensor(wav)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0).unsqueeze(0)  # [1, 1, L]
    elif wav.ndim == 2:
        if wav.shape[0] in (1, 2) and wav.shape[1] > 2:
            wav = wav.unsqueeze(0)  # [1, C, L]
        elif wav.shape[1] in (1, 2) and wav.shape[0] > 2:
            wav = wav.t().unsqueeze(0)  # [1, C, L]
        else:
            wav = wav.unsqueeze(0)
    elif wav.ndim == 3:
        if wav.shape[1] > 2 and wav.shape[2] in (1, 2):
            wav = wav.movedim(-1, 1)  # [B, C, L]

    if wav.shape[1] == 1:
        wav = wav.repeat(1, 2, 1)
    elif wav.shape[1] > 2:
        wav = wav[:, :2, :]
    return wav[:1]


def _encode_ref_audio(audio_vae, audio):
    """Encode audio into H3 audio VAE latent [1, 32, 2, T]."""
    if audio_vae is None:
        raise ValueError("Audio VAE is required when audio clips are present on the timeline")
    waveform = _normalize_audio_waveform(audio.get("waveform"))
    if waveform is None:
        raise ValueError("Audio clip contains no waveform")
    sr = int(audio.get("sample_rate", 32000))
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sr != vae_sr:
        if torchaudio is None:
            raise RuntimeError("torchaudio is required for audio resampling")
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    if waveform.shape[-1] < 800:
        waveform = torch.nn.functional.pad(waveform, (0, 800 - waveform.shape[-1]))
    z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
    return z, int(z.shape[-1])


def _slice_audio(audio, start_sec=0.0, duration_sec=None):
    """Slice an AUDIO dict [B, C, L] by start offset and optional duration."""
    if audio is None:
        return None
    waveform = audio.get("waveform")
    if waveform is None:
        return audio
    waveform = _normalize_audio_waveform(waveform)
    sr = int(audio.get("sample_rate", 32000))
    total_samples = int(waveform.shape[-1])
    start_sample = max(0, int(round(start_sec * sr)))
    if duration_sec is not None and duration_sec > 0:
        end_sample = min(total_samples, start_sample + int(round(duration_sec * sr)))
    else:
        end_sample = total_samples
    if start_sample >= total_samples:
        start_sample = max(0, total_samples - 1)
    sliced = waveform[..., start_sample:end_sample]
    if sliced.shape[-1] == 0:
        sliced = waveform[..., -1:]
    return {"waveform": sliced, "sample_rate": sr}


def _media_ref_path(media):
    """Resolve an uploaded media ref {name, subfolder, type} to a file path."""
    name = media.get("name") if isinstance(media, dict) else None
    if not name:
        raise ValueError("Timeline: clip media ref has no name")
    sub = media.get("subfolder") or ""
    ref = "%s/%s" % (sub, name) if sub else name
    path = folder_paths.get_annotated_filepath("%s [%s]" % (ref, media.get("type") or "input"))
    if not os.path.isfile(path):
        raise ValueError("Timeline: media file not found: %s" % ref)
    return path


def _load_image_file(media):
    """Load an uploaded still image as [1,H,W,C] float IMAGE frames."""
    img = node_helpers.pillow(
        ImageOps.exif_transpose,
        node_helpers.pillow(Image.open, _media_ref_path(media)))
    img = img.convert("RGB")
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)[None]


def _resample_video_frames(collected_timed_frames, target_fps):
    """Resample a list of (timestamp_seconds, frame_tensor) to target_fps."""
    if not collected_timed_frames:
        return None
    times = np.array([item[0] for item in collected_timed_frames])
    frames = [item[1] for item in collected_timed_frames]
    if target_fps is None or target_fps <= 0 or len(frames) == 1 or times[-1] <= times[0]:
        return torch.stack(frames)

    duration = times[-1] - times[0]
    num_target = max(1, int(round(duration * target_fps)) + 1)
    target_times = np.arange(num_target) / float(target_fps) + times[0]
    target_times = target_times[target_times <= times[-1] + 0.5 / float(target_fps)]

    pos = np.clip(np.searchsorted(times, target_times), 1, len(times) - 1)
    left_dist = np.abs(target_times - times[pos - 1])
    right_dist = np.abs(target_times - times[pos])
    indices = np.where(left_dist <= right_dist, pos - 1, pos)
    return torch.stack([frames[i] for i in indices])


def _load_media_file(media, fps=None):
    """Decode an uploaded audio/video file with PyAV.

    Returns {"frames": [B,H,W,C]|None, "audio": AUDIO dict|None}.
    """
    frames, audio = None, None
    mpath = _media_ref_path(media)
    try:
        with av.open(mpath) as af:
            streams = {s.type: s for s in af.streams}
            vstream = streams.get("video")
            vtimebase = float(vstream.time_base) if vstream and vstream.time_base is not None else None
            vrate = float(vstream.average_rate) if vstream and vstream.average_rate else 24.0

            collected, chunks = [], []
            for packet in af.demux():
                stype = packet.stream.type if packet.stream is not None else None
                if stype not in streams:
                    continue
                if stype == "video":
                    for frame in packet.decode():
                        arr = frame.to_ndarray(format="rgb24")
                        if frame.pts is not None and vtimebase is not None:
                            t_sec = float(frame.pts * vtimebase)
                        else:
                            t_sec = float(len(collected) / vrate)
                        collected.append((t_sec, torch.from_numpy(
                            np.asarray(arr, dtype=np.float32) / 255.0)))
                elif stype == "audio":
                    n_channels = streams["audio"].channels
                    for frame in packet.decode():
                        buf = torch.from_numpy(frame.to_ndarray())
                        if buf.shape[0] != n_channels:
                            buf = buf.view(-1, n_channels).t()
                        chunks.append(buf)
            if collected:
                collected.sort(key=lambda item: item[0])
                if fps is not None and fps > 0:
                    frames = _resample_video_frames(collected, fps)
                else:
                    frames = torch.stack([item[1] for item in collected])
            if chunks:
                astream = streams["audio"]
                audio = {
                    "waveform": f32_pcm(torch.cat(chunks, dim=1)).unsqueeze(0),
                    "sample_rate": int(astream.codec_context.sample_rate),
                }
    except Exception as e:
        _LOG.warning("Timeline: PyAV decode failed for %s: %s", mpath, e)

    if frames is None and mpath.lower().endswith(".gif"):
        try:
            with Image.open(mpath) as img:
                collected = []
                t_sec = 0.0
                for frame in ImageSequence.Iterator(img):
                    duration = frame.info.get("duration", 100) / 1000.0
                    arr = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
                    collected.append((t_sec, torch.from_numpy(arr)))
                    t_sec += duration
                if collected:
                    if fps is not None and fps > 0:
                        frames = _resample_video_frames(collected, fps)
                    else:
                        frames = torch.stack([item[1] for item in collected])
        except Exception as e:
            _LOG.warning("Timeline: PIL GIF fallback failed for %s: %s", mpath, e)

    return {"frames": frames, "audio": audio}


class _DynamicInputs(dict):
    """Dynamic backend input map: accepts any key under any declared prefix."""

    def __init__(self, *pairs, **fixed):
        super().__init__(**fixed)
        if len(pairs) == 2 and isinstance(pairs[0], str) and isinstance(pairs[1], (tuple, list)):
            pairs = [pairs]
        self._prefixes = sorted(pairs, key=lambda p: -len(p[0]))

    def __contains__(self, key):
        return super().__contains__(key) or (isinstance(key, str) and any(key.startswith(p[0]) for p in self._prefixes))

    def __getitem__(self, key):
        if isinstance(key, str):
            for p, types in self._prefixes:
                if key.startswith(p):
                    return types
        return super().__getitem__(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class MiniMaxH3Timeline:
    """Video-editor timeline: still images, video clips and audio clips anchored
    along the MiniMax H3 timeline using core minimax_keyframes guides."""

    MAX_CLIPS = 32

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "H3 positive conditioning to attach timeline guides to."}),
                "video vae": ("VAE", {
                    "tooltip": "MiniMax H3 video VAE used to encode images and video clips."}),
                "audio vae": ("VAE", {
                    "tooltip": "H3 audio VAE. Required when any clip on the timeline has audio."}),
                "latent": ("LATENT", {
                    "tooltip": "Target MiniMax H3 AV latent; defines canvas resolution and duration."}),
                "timeline_state": ("STRING", {
                    "default": '{"clips":[]}',
                    "multiline": False,
                    "tooltip": "Internal UI state from the timeline widget."}),
                "crop": (["disabled", "center"], {"default": "disabled"}),
            },
            "optional": _DynamicInputs(
                ("video_", ("IMAGE",)),
                ("video_audio_", ("AUDIO",)),
                ("image_", ("IMAGE",)),
                ("audio_", ("AUDIO",)),
                fps=("INT", {
                    "default": 24, "min": 1, "max": 240, "step": 1,
                    "tooltip": "Timeline frame rate for audio synchronization."}),
                total_frames=("INT", {
                    "default": 240, "min": 1, "max": 100000, "step": 1,
                    "tooltip": "Timeline ruler length in frames."}),
            ),
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Video-editor timeline: place still images, video clips and audio "
                   "tracks at any frame using ComfyUI core MiniMax H3 guides.")

    def apply(self, conditioning, latent, timeline_state, crop="disabled", **kwargs):
        vae = kwargs.get("video vae")
        audio_vae = kwargs.get("audio vae")
        fps = int(kwargs.get("fps") or FPS)

        try:
            state = json.loads(timeline_state or "{}")
        except Exception as exc:
            raise ValueError("Timeline: invalid timeline UI state JSON") from exc

        clips = state.get("clips") or []
        if not clips:
            return (conditioning,)

        if len(clips) > self.MAX_CLIPS:
            raise ValueError("Timeline: clip count exceeds maximum of %d" % self.MAX_CLIPS)

        samples = latent["samples"]
        if hasattr(samples, "tensors"):
            video = samples.tensors[0]
            audio = samples.tensors[1] if len(samples.tensors) > 1 else None
        elif hasattr(samples, "unbind"):
            unbound = samples.unbind()
            video = unbound[0]
            audio = unbound[1] if len(unbound) > 1 else None
        elif isinstance(samples, (list, tuple)):
            video = samples[0]
            audio = samples[1] if len(samples) > 1 else None
        else:
            video = samples
            audio = None

        if video.ndim == 4:
            video = video.unsqueeze(0)

        height = int(video.shape[3]) * 16
        width = int(video.shape[4]) * 16
        latent_t = int(video.shape[2])
        frame_count = _pixel_frames(latent_t)
        audio_latent_t = int(audio.shape[-1]) if audio is not None else int(round(frame_count * FRAME_RESCALE))

        keyframes = list(conditioning[0][1].get("minimax_keyframes", []))

        for idx, clip in enumerate(clips, 1):
            slot = int(clip.get("id") or idx)
            kind = clip.get("kind")
            start = max(1, int(clip.get("start") or 1))
            resolved_frame_index = start - 1

            if kind == "image":
                image = kwargs.get("image_%d" % slot)
                fmedia = clip.get("file")
                if fmedia:
                    image = _load_image_file(fmedia)
                if image is None:
                    continue

                resolved_frame_index = min(frame_count - 1, resolved_frame_index)
                frames = _resize(image[:1], width, height, crop)
                enc_latent = vae.encode(frames)
                keyframes.append({
                    "resolved_frame_index": resolved_frame_index,
                    "latent": enc_latent,
                })

            elif kind == "video":
                frames = kwargs.get("video_%d" % slot)
                audio_in = kwargs.get("video_audio_%d" % slot)
                src_start = max(0, int(clip.get("src_start") or 0))
                fmedia = clip.get("file")
                if fmedia:
                    data = _load_media_file(fmedia, fps=fps)
                    frames = data["frames"]
                    if audio_in is None and not clip.get("audio_off") and data["audio"] is not None:
                        audio_in = _slice_audio(data["audio"], src_start / float(fps))
                elif frames is not None and src_start > 0:
                    if src_start >= int(frames.shape[0]):
                        src_start = max(0, int(frames.shape[0]) - 1)
                    frames = frames[src_start:]
                    if audio_in is not None and not clip.get("audio_off"):
                        audio_in = _slice_audio(audio_in, src_start / float(fps))

                if frames is None:
                    continue

                want = min(max(1, int(clip.get("len") or 22)), int(frames.shape[0]))
                if want < 5:
                    guide_frames = 1
                else:
                    guide_frames = want
                    while guide_frames % 17 != 5:
                        guide_frames -= 1

                resolved_frame_index = max(0, min(frame_count - guide_frames, resolved_frame_index))
                enc_latent = vae.encode(_resize(frames[:guide_frames], width, height, crop))
                kf = {
                    "resolved_frame_index": resolved_frame_index,
                    "latent": enc_latent,
                }

                if audio_in is not None and not clip.get("audio_off"):
                    if clip.get("audio_link", True):
                        audio_slice = _slice_audio(audio_in, 0, guide_frames / float(fps))
                        audio_lat, audio_rt = _encode_ref_audio(audio_vae, audio_slice)
                        max_rt = max(1, math.floor(audio_latent_t - FRAME_RESCALE * resolved_frame_index))
                        if audio_rt > max_rt:
                            audio_lat = audio_lat[..., :max_rt].clone()
                        kf["audio_latent"] = audio_lat
                        keyframes.append(kf)
                    else:
                        keyframes.append(kf)
                        a_start = int(clip.get("audio_start") or start)
                        a_resolved = max(0, min(frame_count - 1, a_start - 1))
                        a_len = int(clip.get("audio_len") or want)
                        asrc = float(clip.get("audio_src_start") or 0)
                        audio_slice = _slice_audio(audio_in, asrc / float(fps), a_len / float(fps))
                        audio_lat, audio_rt = _encode_ref_audio(audio_vae, audio_slice)
                        max_rt = max(1, math.floor(audio_latent_t - FRAME_RESCALE * a_resolved))
                        if audio_rt > max_rt:
                            audio_lat = audio_lat[..., :max_rt].clone()
                        keyframes.append({
                            "resolved_frame_index": a_resolved,
                            "audio_latent": audio_lat,
                        })
                else:
                    keyframes.append(kf)

            elif kind == "audio":
                audio_in = kwargs.get("audio_%d" % slot)
                src_start = max(0, int(clip.get("src_start") or 0))
                fmedia = clip.get("file")
                if fmedia:
                    data = _load_media_file(fmedia)
                    audio_in = data["audio"]
                if audio_in is None:
                    continue

                want_len = max(1, int(clip.get("len") or 22))
                resolved_frame_index = max(0, min(frame_count - 1, resolved_frame_index))
                audio_slice = _slice_audio(audio_in, src_start / float(fps), want_len / float(fps))
                audio_lat, audio_rt = _encode_ref_audio(audio_vae, audio_slice)
                max_rt = max(1, math.floor(audio_latent_t - FRAME_RESCALE * resolved_frame_index))
                if audio_rt > max_rt:
                    audio_lat = audio_lat[..., :max_rt].clone()
                keyframes.append({
                    "resolved_frame_index": resolved_frame_index,
                    "audio_latent": audio_lat,
                })

        keyframes.sort(key=lambda k: k.get("resolved_frame_index", 0))
        out = node_helpers.conditioning_set_values(conditioning, {"minimax_keyframes": keyframes})
        return (out,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3Timeline": MiniMaxH3Timeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3Timeline": "H3 Timeline Editor",
}
