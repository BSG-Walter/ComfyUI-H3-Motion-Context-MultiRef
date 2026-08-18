"""MiniMax H3 Timeline Editor.

Visual video-editor timeline node for MiniMax H3, using native ComfyUI core
minimax_keyframes guides and native per-token denoise masks (commit
ff6c8a8a, PR #15375): clip content is injected as clean latent + noise_mask
into the sampler, replacing the old post-CFG hard-clamp hook.
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
import comfy.nested_tensor
import comfy.utils
from comfy_extras.nodes_audio import f32_pcm
from PIL import Image, ImageOps, ImageSequence

try:
    import torchaudio
except ImportError:
    torchaudio = None

_LOG = logging.getLogger("h3_timeline")
DEBUG_LOGS = False

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
FPS = 24


def _pixel_frames(latent_t):
    """Pixel frames covered by latent_t latent steps."""
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))


def _video_token_window(lead, want):
    """Segment-latent token range intersecting content frames [lead, lead+want).

    Tokens follow the VAE's (1,4,4,4,4) temporal grid: token i covers
    [cum, cum+w) segment frames. The VAE encodes each 17-frame chunk
    independently, so tokens map 1:1 onto the target video's latent grid
    when the segment starts on the 17-frame boundary.
    """
    seg_cum = 0
    i0, i1 = None, 0
    for i in range(1024):
        w = FRAME_PER_TOKEN[i % 5]
        if i0 is None and seg_cum < lead + want and seg_cum + w > lead:
            i0 = i
        if seg_cum >= lead + want:
            i1 = i
            break
        seg_cum += w
    if i0 is None:
        i0 = 0
    if i1 == 0:
        i1 = i0 + 1
    return i0, i1


def _grid_segment(frame_idx, want, frame_count):
    """Grid-align timeline content [frame_idx, frame_idx+want) for VAE clamping.

    Returns (seg_start, lead, seg_len): the containing 17-grid boundary, the
    lead-pad count, and the padded segment length (valid 17k+5 shape) covering
    the content without exceeding the video's final boundary.
    """
    lead = frame_idx % 17
    seg_start = frame_idx - lead
    span = want + lead
    if span <= 5:
        seg_len = 5
    else:
        seg_len = 17 * ((span - 5 + 16) // 17) + 5
    budget = frame_count - seg_start
    if seg_len > budget:
        seg_len = ((budget - 5) // 17) * 17 + 5
        if seg_len < span:
            seg_len = span
    return seg_start, lead, seg_len


def _pad_clip(frames, lead, tail):
    """Pad frames [F,H,W,C] with repeated edge frames: lead before, tail after."""
    if lead == 0 and tail == 0:
        return frames
    parts = []
    if lead > 0:
        parts.append(frames[:1].repeat(lead, 1, 1, 1))
    parts.append(frames)
    if tail > 0:
        parts.append(frames[-1:].repeat(tail, 1, 1, 1))
    return torch.cat(parts, dim=0)


def _frame_to_audio_latent_idx(frame_idx, fps=FPS):
    """Map pixel frame index to audio latent token index at 40 latents/sec."""
    if frame_idx <= 0:
        return 0
    sec = float(frame_idx) / float(fps or FPS)
    return int(round(sec * 40.0))


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
    """Encode audio into H3 audio VAE latent [1, 32, 2, T] using comfy-core VAE encode."""
    if audio_vae is None:
        raise ValueError("Audio VAE is required when audio clips are present on the timeline")
    waveform = audio.get("waveform")
    if waveform is None:
        raise ValueError("Audio clip contains no waveform")
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    
    sr = int(audio.get("sample_rate", 32000))
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sr != vae_sr:
        if torchaudio is None:
            raise RuntimeError("torchaudio is required for audio resampling")
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)

    # VAE.encode expects input with channel dim at the end: [B, length, 2] stereo
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim == 3 and waveform.shape[1] in (1, 2) and waveform.shape[2] > 2:
        # [B, C, L] -> [B, L, C]
        if waveform.shape[1] == 1:
            waveform = waveform.repeat(1, 2, 1)
        waveform = waveform.movedim(1, -1)

    if waveform.ndim < 3 or waveform.shape[-1] == 0 or waveform.shape[-2] == 0:
        _LOG.warning("Timeline: skipping degenerate audio waveform shape=%s sr=%d", tuple(waveform.shape), sr)
        return None, 0

    z = audio_vae.encode(waveform)  # Returns [1, 32, 2, T]
    return z, int(z.shape[-1])


def _slice_audio(audio, start_sec=0.0, duration_sec=None):
    """Slice an AUDIO dict [B, C, L] by start offset and optional duration.

    Returns None when the source has no usable audio (0 channels or 0 samples),
    so callers can skip the clip instead of feeding a degenerate waveform to the VAE.
    """
    if audio is None:
        return None
    waveform = audio.get("waveform")
    if waveform is None:
        return audio
    waveform = _normalize_audio_waveform(waveform)
    if waveform.shape[-1] == 0 or waveform.shape[-2] == 0:
        return None
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


def _prepare_audio_latent(audio_vae, audio, src_sec, duration_sec, resolved_frame, audio_latent_t):
    """Slice, encode, and clamp audio latent to the timeline boundary."""
    audio_slice = _slice_audio(audio, src_sec, duration_sec)
    if audio_slice is None:
        _LOG.warning("Timeline: audio slice empty (src_sec=%.3f duration=%.3f frame=%d)", src_sec, duration_sec, resolved_frame)
        return None
    audio_lat, audio_rt = _encode_ref_audio(audio_vae, audio_slice)
    if audio_lat is None:
        return None
    max_rt = max(1, math.floor(audio_latent_t - FRAME_RESCALE * resolved_frame))
    return audio_lat[..., :max_rt].clone() if audio_rt > max_rt else audio_lat


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


def _load_image_file(media, fps=None):
    """Load an uploaded image or animated GIF/webp as [F,H,W,C] float IMAGE frames."""
    path = _media_ref_path(media)
    try:
        with Image.open(path) as img:
            n_frames = getattr(img, "n_frames", 1)
            if n_frames > 1:
                collected = []
                t_sec = 0.0
                for frame in ImageSequence.Iterator(img):
                    duration = frame.info.get("duration", 100) / 1000.0
                    arr = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
                    collected.append((t_sec, torch.from_numpy(arr)))
                    t_sec += duration
                if collected:
                    if fps is not None and fps > 0:
                        return _resample_video_frames(collected, fps)
                    return torch.stack([item[1] for item in collected])
            else:
                img_conv = ImageOps.exif_transpose(img).convert("RGB")
                return torch.from_numpy(np.asarray(img_conv, dtype=np.float32) / 255.0)[None]
    except Exception as e:
        _LOG.warning("Timeline: _load_image_file failed for %s: %s", path, e)
        return None


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


def _load_media_file(media, fps=None, start_sec=0.0, duration_sec=None, load_video=True, load_audio=True):
    """Decode an uploaded audio/video file with PyAV.

    Only decodes the requested time window [start_sec, start_sec + duration_sec] for speed.
    Returns {"frames": [B,H,W,C]|None, "audio": AUDIO dict|None}.
    """
    frames, audio = None, None
    mpath = _media_ref_path(media)
    start_sec = max(0.0, float(start_sec or 0.0))
    duration_sec = float(duration_sec) if duration_sec is not None and duration_sec > 0 else None

    try:
        with av.open(mpath) as af:
            streams = {s.type: s for s in af.streams}
            vstream = streams.get("video") if load_video else None
            astream = streams.get("audio") if load_audio else None

            vtimebase = float(vstream.time_base) if vstream and vstream.time_base is not None else None
            vrate = float(vstream.average_rate) if vstream and vstream.average_rate else (float(fps) if fps else 24.0)
            atimebase = (float(astream.time_base)
                         if astream and astream.time_base is not None
                         else (1.0 / float(astream.codec_context.sample_rate) if astream else None))

            if vstream is not None and start_sec > 0:
                try:
                    seek_target = max(0.0, start_sec - 1.0)
                    if vtimebase:
                        seek_pts = int(round(seek_target / vtimebase))
                        af.seek(seek_pts, backward=True, stream=vstream)
                    else:
                        af.seek(int(round(seek_target * av.time_base)), backward=True)
                except Exception as e:
                    _LOG.debug("Timeline: seek failed, decoding sequentially: %s", e)

            target_streams = [s for s in (vstream, astream) if s is not None]
            if not target_streams:
                return {"frames": None, "audio": None}

            collected, chunks = [], []
            end_sec = (start_sec + duration_sec + 1.0 / (fps or vrate)) if duration_sec is not None else None
            needed_count = int(math.ceil(duration_sec * (fps or vrate))) if duration_sec is not None else 0

            for packet in af.demux(target_streams):
                stype = packet.stream.type if packet.stream is not None else None
                if stype == "video" and vstream is not None:
                    for frame in packet.decode():
                        if frame.pts is not None and vtimebase is not None:
                            t_sec = float(frame.pts * vtimebase)
                        else:
                            t_sec = float(len(collected) / vrate) + start_sec

                        if t_sec < start_sec - (0.5 / (fps or vrate)):
                            continue
                        if end_sec is not None and t_sec > end_sec and len(collected) >= needed_count:
                            break

                        arr = frame.to_ndarray(format="rgb24")
                        collected.append((t_sec, torch.from_numpy(
                            np.asarray(arr, dtype=np.float32) / 255.0)))
                    if end_sec is not None and collected and collected[-1][0] > end_sec and len(collected) >= needed_count:
                        if astream is None or chunks:
                            break
                elif stype == "audio" and astream is not None:
                    n_channels = astream.channels
                    sr = float(astream.codec_context.sample_rate)
                    for frame in packet.decode():
                        # Time-filter like the video stream: the seek lands ~1s
                        # early (tolerance pre-roll), so raw audio would start
                        # at the wrong file position. Keep only frames that
                        # overlap [start_sec, end_sec).
                        if frame.pts is not None and atimebase:
                            t_sec = float(frame.pts * atimebase)
                            f_dur = frame.samples / sr
                            if end_sec is not None and t_sec > end_sec + f_dur:
                                continue
                            if t_sec + f_dur < start_sec:
                                continue
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
            if chunks and astream is not None:
                audio = {
                    "waveform": f32_pcm(torch.cat(chunks, dim=1)).unsqueeze(0),
                    "sample_rate": int(astream.codec_context.sample_rate),
                }
    except Exception as e:
        _LOG.warning("Timeline: PyAV decode failed for %s: %s", mpath, e)

    if frames is None and load_video and mpath.lower().endswith(".gif"):
        frames = _load_image_file(media, fps=fps)

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
                    "default": 243, "min": 5, "max": 100000, "step": 17,
                    "tooltip": "Timeline ruler length in frames (valid 5 + 17N frames: 5, 22, 39, 56, ...)."}),
                clamp_strength=("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Mask preservation strength. 1.0 = exact source preservation (denoise mask 0), <1.0 = blend with DiT prediction (mask > 0)."}),
            ),
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT",)
    RETURN_NAMES = ("conditioning", "latent",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Video-editor timeline: place still images, video clips and audio "
                   "tracks at any frame using ComfyUI core MiniMax H3 guides.")

    def apply(self, conditioning, latent, timeline_state, crop="disabled", **kwargs):
        clamp_strength = float(kwargs.get("clamp_strength", 1.0))
        vae = kwargs.get("video vae")
        audio_vae = kwargs.get("audio vae")
        fps = int(kwargs.get("fps") or FPS)

        try:
            state = json.loads(timeline_state or "{}")
        except Exception as exc:
            raise ValueError("Timeline: invalid timeline UI state JSON") from exc

        clips = state.get("clips") or []

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
        clean_video = torch.zeros_like(video)
        video_mask = torch.ones([1, 1] + list(video.shape[2:]), dtype=torch.float32, device=video.device)
        clean_audio = torch.zeros_like(audio) if audio is not None else None
        audio_mask = (torch.ones([1, 1] + list(audio.shape[2:]), dtype=torch.float32, device=audio.device)
                      if audio is not None else None)
        audio_jobs = []

        def preserve(span_start, span_end, span_latent, strength, is_audio=False):
            """Inject an encoded clip window into the clean composite + mask."""
            strength = max(0.0, min(1.0, float(strength) * float(clamp_strength)))
            mask_val = 1.0 - strength
            if is_audio:
                span_end = min(int(audio_mask.shape[-1]), span_end)
                if span_end <= span_start:
                    return
                clean_audio[..., span_start:span_end] = span_latent[..., :span_end - span_start]
                audio_mask[..., span_start:span_end] = torch.minimum(
                    audio_mask[..., span_start:span_end],
                    torch.full_like(audio_mask[..., span_start:span_end], mask_val))
            else:
                span_end = min(int(video_mask.shape[2]), span_end)
                if span_end <= span_start:
                    return
                clean_video[:, :, span_start:span_end] = span_latent
                video_mask[:, :, span_start:span_end] = torch.minimum(
                    video_mask[:, :, span_start:span_end],
                    torch.full_like(video_mask[:, :, span_start:span_end], mask_val))

        for idx, clip in enumerate(clips, 1):
            slot = int(clip.get("id") or idx)
            kind = clip.get("kind")
            start = max(1, int(clip.get("start") or 1))
            resolved_frame_index = start - 1

            if resolved_frame_index >= frame_count:
                continue

            if kind == "image":
                image = kwargs.get("image_%d" % slot)
                src_start = max(0, int(clip.get("src_start") or 0))
                fmedia = clip.get("file")
                if fmedia:
                    image = _load_image_file(fmedia, fps=fps)
                if image is None:
                    continue

                if src_start > 0 and image.shape[0] > 1:
                    if src_start >= int(image.shape[0]):
                        src_start = max(0, int(image.shape[0]) - 1)
                    image = image[src_start:]

                want_len = max(1, min(int(clip.get("len") or 1), frame_count - resolved_frame_index))

                if image.shape[0] > 1:
                    content = _resize(image[:want_len], width, height, crop)
                    if int(content.shape[0]) < want_len:
                        last_f = content[-1:]
                        content = torch.cat([content, last_f.repeat(want_len - int(content.shape[0]), 1, 1, 1)], dim=0)
                else:
                    base_frame = _resize(image[:1], width, height, crop)
                    content = base_frame.repeat(want_len, 1, 1, 1)

                seg_start, lead, seg_len = _grid_segment(resolved_frame_index, want_len, frame_count)
                seg = _pad_clip(content, lead, seg_len - want_len - lead)
                enc_latent = vae.encode(seg)
                i0, i1 = _video_token_window(lead, want_len)
                if i1 > int(enc_latent.shape[2]):
                    i1 = int(enc_latent.shape[2])
                if i0 >= i1:
                    i0 = max(0, i1 - 1)
                win_latent = enc_latent[:, :, i0:i1]

                keyframes.append({
                    "resolved_frame_index": resolved_frame_index,
                    "latent": win_latent,
                })
                preserve((seg_start // 17) * 5 + i0, (seg_start // 17) * 5 + i1,
                         enc_latent[:, :, i0:i1], clip.get("strength", 1.0))

            elif kind == "video":
                frames = kwargs.get("video_%d" % slot)
                audio_in = kwargs.get("video_audio_%d" % slot)
                src_start = max(0, int(clip.get("src_start") or 0))
                fmedia = clip.get("file")

                # Clamp clip strictly to timeline boundary
                avail_timeline_frames = frame_count - resolved_frame_index
                if avail_timeline_frames <= 0:
                    continue

                clip_len = max(1, int(clip.get("len") or 22))
                want = min(clip_len, avail_timeline_frames)
                if frames is not None and not fmedia:
                    want = min(want, max(1, int(frames.shape[0]) - src_start))

                data = None
                if fmedia:
                    start_sec = src_start / float(fps)
                    duration_sec = clip_len / float(fps)
                    data = _load_media_file(
                        fmedia,
                        fps=fps,
                        start_sec=start_sec,
                        duration_sec=duration_sec,
                        load_video=True,
                        load_audio=not clip.get("audio_off") and audio_in is None,
                    )
                    frames = data["frames"]
                    if audio_in is None and not clip.get("audio_off") and data["audio"] is not None:
                        audio_in = data["audio"]
                elif frames is not None and src_start > 0:
                    if src_start >= int(frames.shape[0]):
                        src_start = max(0, int(frames.shape[0]) - 1)
                    frames = frames[src_start:]
                    if audio_in is not None and not clip.get("audio_off"):
                        audio_in = _slice_audio(audio_in, src_start / float(fps))

                if frames is None:
                    continue

                content = frames[:want]
                if int(content.shape[0]) < want:
                    last_f = content[-1:]
                    content = torch.cat([content, last_f.repeat(want - int(content.shape[0]), 1, 1, 1)], dim=0)

                # One VAE encode per clip: grid-align the segment to the VAE's
                # independent 17-frame chunk lattice so tokens map 1:1 onto the
                # target latent. Lead/tail are frozen repeats; only the tokens
                # intersecting the clip are clamped and reused as the
                # cross-attention guide (a constant <=4 frame offset, invisible
                # under soft guidance).
                seg_start, lead, seg_len = _grid_segment(resolved_frame_index, want, frame_count)
                seg = _pad_clip(_resize(content, width, height, crop), lead, seg_len - want - lead)
                enc_latent = vae.encode(seg)
                i0, i1 = _video_token_window(lead, want)
                if i1 > int(enc_latent.shape[2]):
                    i1 = int(enc_latent.shape[2])
                if i0 >= i1:
                    i0 = max(0, i1 - 1)
                win_latent = enc_latent[:, :, i0:i1]

                # Cross-attention conditioning keyframe (same latent as the injection)
                kf = {
                    "resolved_frame_index": resolved_frame_index,
                    "latent": win_latent,
                }

                # Native denoise-mask injection
                preserve((seg_start // 17) * 5 + i0, (seg_start // 17) * 5 + i1,
                         enc_latent[:, :, i0:i1], clip.get("strength", 1.0))

                if audio_in is not None and not clip.get("audio_off") and clip.get("audio_link", True):
                    # Audio encoding deferred to the second pass (all video encodes first)
                    audio_jobs.append({
                        "kf": kf,
                        "audio": audio_in,
                        "src_sec": 0.0,
                        "duration": want / float(fps),
                        "frame": resolved_frame_index,
                        "strength": float(clip.get("audio_strength", clip.get("strength", 1.0))),
                    })
                keyframes.append(kf)

                if audio_in is not None and not clip.get("audio_off") and not clip.get("audio_link", True):
                    a_start = int(clip.get("audio_start") or start)
                    a_resolved = a_start - 1
                    if a_resolved < frame_count:
                        a_avail = frame_count - a_resolved
                        a_len = min(int(clip.get("audio_len") or want), a_avail)
                        asrc = float(clip.get("audio_src_start") if clip.get("audio_src_start") is not None else src_start)
                        raw_audio = data["audio"] if fmedia and data is not None and data.get("audio") is not None else audio_in
                        src_sec = (asrc - src_start) / float(fps)

                        a_kf = {"resolved_frame_index": a_resolved}
                        keyframes.append(a_kf)
                        audio_jobs.append({
                            "kf": a_kf,
                            "audio": raw_audio,
                            "src_sec": src_sec,
                            "duration": a_len / float(fps),
                            "frame": a_resolved,
                            "strength": float(clip.get("audio_strength", clip.get("strength", 1.0))),
                        })

            elif kind == "audio":
                audio_in = kwargs.get("audio_%d" % slot)
                src_start = max(0, int(clip.get("src_start") or 0))
                fmedia = clip.get("file")
                if fmedia:
                    data = _load_media_file(fmedia, load_video=False, load_audio=True)
                    audio_in = data["audio"]
                if audio_in is None or resolved_frame_index >= frame_count:
                    continue

                want_len = min(max(1, int(clip.get("len") or 22)), frame_count - resolved_frame_index)
                a_kf = {"resolved_frame_index": resolved_frame_index}
                keyframes.append(a_kf)
                audio_jobs.append({
                    "kf": a_kf,
                    "audio": audio_in,
                    "src_sec": src_start / float(fps),
                    "duration": want_len / float(fps),
                    "frame": resolved_frame_index,
                    "strength": float(clip.get("strength", 1.0)),
                })

        # Second pass: all audio encodes together so the audio VAE is loaded once
        # (the video VAE stayed resident through the first pass).
        for job in audio_jobs:
            audio_lat = _prepare_audio_latent(
                audio_vae, job["audio"], job["src_sec"], job["duration"], job["frame"], audio_latent_t
            )
            if audio_lat is None:
                continue
            job["kf"]["audio_latent"] = audio_lat
            preserve(_frame_to_audio_latent_idx(job["frame"], fps=fps),
                     _frame_to_audio_latent_idx(job["frame"], fps=fps) + int(audio_lat.shape[-1]),
                     audio_lat, job["strength"], is_audio=True)

        keyframes.sort(key=lambda k: k.get("resolved_frame_index", 0))
        if DEBUG_LOGS:
            masked_v = int((video_mask < 1.0).sum().item())
            masked_a = int((audio_mask < 1.0).sum().item()) if audio_mask is not None else 0
            print(f"[H3 Timeline] Compiled {len(keyframes)} keyframes, mask coverage: {masked_v} video tokens, {masked_a} audio tokens:")
            for idx, kf in enumerate(keyframes):
                has_v = "latent" in kf
                has_a = "audio_latent" in kf
                rf = kf.get("resolved_frame_index")
                print(f"  - Keyframe #{idx}: frame_idx={rf}, video={has_v}, audio={has_a}")

        out_cond = node_helpers.conditioning_set_values(conditioning, {"minimax_keyframes": keyframes})
        if audio is not None:
            samples = comfy.nested_tensor.NestedTensor((clean_video, clean_audio))
            noise_mask = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
        else:
            samples = clean_video
            noise_mask = video_mask
        out_latent = {"samples": samples, "noise_mask": noise_mask}

        return (out_cond, out_latent)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3Timeline": MiniMaxH3Timeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3Timeline": "H3 Timeline Editor",
}
