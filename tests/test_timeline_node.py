"""Runtime checks for the Timeline super node (images + videos + audios).

Needs ComfyUI's own python (torch + comfy.ldm.minimax importable), i.e. the
.venv next to ComfyUI, not the bare python on PATH. No GPU required.

    & ComfyUI/.venv/Scripts/python.exe tests/test_timeline_node.py

Covers: mixed-clip state parsing, linked vs unlinked video audio, audio
window head/tail slicing, out-of-range audio clamping, and structural
fits that must raise.
"""

import importlib.util
import sys
from pathlib import Path

import torch

COMFY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMFY_ROOT))

base = "custom_nodes.ComfyUI-H3-Motion-Context-MultiRef"
node_dir = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    base, str(node_dir / "__init__.py"),
    submodule_search_locations=[str(node_dir)])
pkg = importlib.util.module_from_spec(spec)
sys.modules[base] = pkg
spec.loader.exec_module(pkg)
n, pl = pkg.nodes, pkg.patch_layout


class FakeVideoVAE:
    def encode(self, pix):
        # real H3 video VAE: 1 frame -> 1 still step, 5 frames -> 2 steps
        steps = 1 if pix.shape[0] == 1 else 2
        return torch.zeros(pix.shape[0], 4, steps, 2, 2)


class FakeStillVAE:
    def encode(self, pix):
        return torch.zeros(pix.shape[0], 4, 1, 2, 2)


class FakeAV:
    def unbind(self):
        # 25 latent steps -> 85 pixel frames (5x [1,4,4,4,4])
        return (torch.zeros(1, 4, 25, 1, 1), torch.zeros(1, 4, 2, 16))


class AudioVAE:
    audio_sample_rate = 16000

    def __init__(self):
        self.windows = []

    def encode(self, wav):
        self.windows.append(wav.clone())
        return torch.zeros(1, 32, 2, 12)


def audio(seconds=1.0, sr=16000):
    n = int(sr * seconds)
    ramp = torch.linspace(0, 1, n).unsqueeze(0).unsqueeze(0)
    return {"waveform": ramp, "sample_rate": sr}


def head(src, samples):
    return src[:, :, :samples].movedim(1, -1)[0]  # what the VAE sees


def tail(src, samples):
    return src[:, :, -samples:].movedim(1, -1)[0]


def run(state, audio_vae=None, vae=None, **inputs):
    node = n.MiniMaxH3Timeline()
    out = node.apply(
        [[torch.zeros(1, 7, 4), {}]],
        {"samples": FakeAV()},
        state,
        "disabled",
        **{"video vae": vae or FakeVideoVAE(), "audio vae": audio_vae},
        **inputs,
    )
    return out[0][0][1]


# --- mixed timeline: image + video (linked audio) + audio clip ---
av = AudioVAE()
state = '{"clips":[' \
        '{"id":1,"kind":"image","start":1,"strength":1},' \
        '{"id":2,"kind":"video","start":10,"strength":0.5,"len":22,' \
        '"audio_link":true},' \
        '{"id":3,"kind":"audio","start":60,"strength":1,"len":22,"align":"head"}]}'
cond = run(state, av, vae=FakeVideoVAE(),
           image_1=torch.rand(1, 12, 12, 3),
           video_2=torch.rand(5, 12, 12, 3),
           video_audio_2=audio(),
           audio_3=audio())
assert cond["minimax_frame_count"] == 85
kfs = cond["minimax_keyframes"]
assert len(kfs) == 3, len(kfs)
assert kfs[0][pl.MC_KEY] == 0 and kfs[0][pl.MC_VIDEO_STRENGTH] == 1.0
assert kfs[1][pl.MC_KEY] == 9 and kfs[1][pl.MC_VIDEO_STRENGTH] == 0.5
assert kfs[2][pl.MC_KEY] == 10 and kfs[2][pl.MC_VIDEO_STRENGTH] == 0.5
refs = cond["minimax_refs"]
assert len(refs) == 2, len(refs)
# linked video audio: same span as the video, same strength
assert refs[0][pl.MC_AUDIO_KEY] == 14.0, refs[0][pl.MC_AUDIO_KEY]
assert refs[0][pl.MC_AUDIO_STRENGTH] == 0.5
assert refs[1][pl.MC_AUDIO_KEY] == 81.0, refs[1][pl.MC_AUDIO_KEY]
assert refs[1][pl.MC_AUDIO_STRENGTH] == 1.0
# linked video audio is windowed from the HEAD (5 frames of sound)
w0 = av.windows[0]
want0 = int(round(5 / 24.0 * 16000))
assert int(w0.shape[1]) == want0, w0.shape
src0 = audio().get("waveform")
assert torch.allclose(w0[0], head(src0, want0))  # first `want0` samples
# audio clip head window: first 22 frames of sound
w1 = av.windows[1]
want1 = int(round(22 / 24.0 * 16000))
assert int(w1.shape[1]) == want1, w1.shape
assert torch.allclose(w1[0], head(src0, want1))
print("mixed timeline OK: image + linked video + audio clip")


# --- unlinked video audio: independent position/length, tail window ---
av2 = AudioVAE()
state = '{"clips":[' \
        '{"id":1,"kind":"video","start":10,"len":22,"audio_link":false,' \
        '"audio_start":30,"audio_len":10,"audio_align":"tail"}]}'
cond = run(state, av2, vae=FakeVideoVAE(),
           video_1=torch.rand(5, 12, 12, 3), video_audio_1=audio())
refs = cond["minimax_refs"]
assert len(refs) == 1
assert refs[0][pl.MC_AUDIO_KEY] == 39.0, refs[0][pl.MC_AUDIO_KEY]  # 30-1+10
w = av2.windows[0]
want = int(round(10 / 24.0 * 16000))
assert int(w.shape[1]) == want
src = audio().get("waveform")  # ramp 0..1
assert torch.allclose(w[0], tail(src, want))  # tail window
print("unlinked video audio OK: independent position + tail window")


# --- linked ignores stale audio_start/audio_len in the state ---
av3 = AudioVAE()
state = '{"clips":[' \
        '{"id":1,"kind":"video","start":5,"len":22,"audio_link":true,' \
        '"audio_start":90,"audio_len":2}]}'
cond = run(state, av3, vae=FakeVideoVAE(),
           video_1=torch.rand(5, 12, 12, 3), video_audio_1=audio())
assert cond["minimax_refs"][0][pl.MC_AUDIO_KEY] == 9.0  # 5-1+5, not 91
print("linked audio ignores stale state OK")


# --- UI clip IDs are stable even after deletes/splits leave gaps ---
avgap = AudioVAE()
state = '{"clips":[' \
        '{"id":7,"kind":"video","start":1,"strength":0.5,"len":5,' \
        '"audio_link":true},' \
        '{"id":9,"kind":"audio","start":20,"strength":1,"len":5}]}'
cond = run(state, avgap, vae=FakeVideoVAE(),
           video_7=torch.rand(5, 12, 12, 3),
           video_audio_7=audio(),
           audio_9=audio())
assert len(cond["minimax_keyframes"]) == 2
assert len(cond["minimax_refs"]) == 2
print("timeline stable clip ids OK")


# --- strength envelope: per-frame video strengths + segmented audio ---
av5 = AudioVAE()
state = '{"clips":[' \
        '{"id":1,"kind":"video","start":10,"len":22,"audio_link":true,' \
        '"env":[[0,0.2],[10,1.0]],"strength":0.7},' \
        '{"id":2,"kind":"audio","start":60,"len":22,' \
        '"env":[[0,1.0],[11,0.5],[21,0.2]]}]}'
cond = run(state, av5, vae=FakeVideoVAE(),
           video_1=torch.rand(5, 12, 12, 3), video_audio_1=audio(),
           audio_2=audio())
kfs = cond["minimax_keyframes"]
assert len(kfs) == 2, len(kfs)
s0, s1 = kfs[0][pl.MC_VIDEO_STRENGTH], kfs[1][pl.MC_VIDEO_STRENGTH]
assert s0 == 0.2, s0  # step 0 at pixel offset 0
assert abs(s1 - 0.28) < 1e-9, s1  # step 1 at offset 1, ramping 0.2 -> 1.0 over 10 frames
refs = cond["minimax_refs"]
# video audio: 5-frame run, the frame-10 cut is outside the window -> one
# flat ref at the envelope's first strength (the flat `strength` is ignored)
assert len(refs) == 4, len(refs)
assert refs[0][pl.MC_AUDIO_KEY] == 14.0, refs[0][pl.MC_AUDIO_KEY]
assert refs[0]["ref_audio_t"] == 12
assert refs[0][pl.MC_AUDIO_STRENGTH] == 0.2
seg = refs[1:]
# audio clip: cuts at frames 11 and 21 split the 12 latent steps 6/5/1
assert [r["ref_audio_t"] for r in seg] == [6, 5, 1], \
    [r["ref_audio_t"] for r in seg]
assert [r[pl.MC_AUDIO_STRENGTH] for r in seg] == [1.0, 0.5, 0.2], \
    [r[pl.MC_AUDIO_STRENGTH] for r in seg]
assert [r[pl.MC_AUDIO_KEY] for r in seg] == [70.0, 80.0, 81.0], \
    [r[pl.MC_AUDIO_KEY] for r in seg]
# the window was encoded once, then sliced: exactly 2 encode calls total
assert len(av5.windows) == 2, len(av5.windows)
# a single-point envelope flattens the clip at that strength
av6 = AudioVAE()
state = '{"clips":[{"id":1,"kind":"video","start":4,"len":22,' \
        '"env":[[7,0.3]]}]}'
cond = run(state, av6, vae=FakeVideoVAE(),
           video_1=torch.rand(5, 12, 12, 3))
assert all(kf[pl.MC_VIDEO_STRENGTH] == 0.3
           for kf in cond["minimax_keyframes"])
print("strength envelope OK: per-frame video strengths + segmented audio")


# --- out-of-range audio clamps with a warning, never raises ---
av4 = AudioVAE()
state = '{"clips":[{"id":1,"kind":"audio","start":200,"len":22}]}'
cond = run(state, av4, vae=FakeVideoVAE(), audio_1=audio())
assert cond["minimax_refs"][0][pl.MC_AUDIO_KEY] == 85.0  # parked at last frame
print("out-of-range audio clamp OK")


# --- audio_off: the deleted audio band must stay silent ---
avoff = AudioVAE()
state = '{"clips":[' \
        '{"id":1,"kind":"video","start":10,"len":22,' \
        '"audio_link":false,"audio_start":30,"audio_len":10,' \
        '"audio_off":true}]}'
cond = run(state, avoff, vae=FakeVideoVAE(),
           video_1=torch.rand(5, 12, 12, 3), video_audio_1=audio())
assert len(cond["minimax_keyframes"]) == 2
assert (cond.get("minimax_refs") or []) == [], cond.get("minimax_refs")
assert len(avoff.windows) == 0  # the VAE never saw the wired audio
print("audio_off video clip OK: band deleted, video stays, silent")


# --- out-of-range video clamps with a warning, never raises ---
cond = run('{"clips":[{"id":1,"kind":"video","start":200,"len":22}]}',
           AudioVAE(), vae=FakeVideoVAE(),
           video_1=torch.rand(5, 12, 12, 3))
assert len(cond["minimax_keyframes"]) == 2
assert cond["minimax_keyframes"][0][pl.MC_KEY] == 80  # parked at frame 81
print("out-of-range video clamp OK")


# --- out-of-range image clamps with a warning, never raises ---
cond = run('{"clips":[{"id":1,"kind":"image","start":90}]}',
           AudioVAE(), vae=FakeVideoVAE(),
           image_1=torch.rand(1, 12, 12, 3))
assert len(cond["minimax_keyframes"]) == 1
assert cond["minimax_keyframes"][0][pl.MC_KEY] == 84  # parked at last frame
print("out-of-range image clamp OK")


# --- structural violations still raise ---
try:
    run('{"clips":[{"id":1,"kind":"bogus"}]}')
    raise AssertionError("unknown kind must raise")
except ValueError:
    pass
print("structural validation OK")


# --- audio clip without audio_vae raises; without audio input raises ---
try:
    run('{"clips":[{"id":1,"kind":"audio","start":5}]}', None,
        vae=FakeVideoVAE())
    raise AssertionError("audio without audio_vae must raise")
except ValueError:
    pass
print("audio_vae requirement OK")


# --- uploaded media files: clips reference {name, subfolder, type} ---
from PIL import Image as _PILImage
import io as _io
import imageio_ffmpeg
import json
import numpy as np
import os
import wave

folder_paths = pkg.nodes.folder_paths
inp_dir = folder_paths.get_input_directory()
os.makedirs(inp_dir, exist_ok=True)

_tmp = []
def _tmpfile(name, write):
    path = os.path.join(inp_dir, name)
    write(path)
    _tmp.append(path)
    return {"name": name, "subfolder": "", "type": "input"}

# still image
_tmpfile("h3tl_tmp.png", lambda p: _PILImage.fromarray(
    np.full((12, 12, 3), 255, dtype=np.uint8)).save(p))
# 16-frame mp4 (no audio track)
def _mk_video(p):
    w = imageio_ffmpeg.write_frames(p, (12, 12), fps=24)
    w.send(None)
    for _ in range(16):
        w.send(np.full((12, 12, 3), 200, dtype=np.uint8))
    w.close()
_tmpfile("h3tl_tmp.mp4", _mk_video)
# 2 s mono wav at 16 kHz
def _mk_wav(p):
    n = 32000
    data = (np.linspace(-1, 1, n) * 32767).astype(np.int16)
    with wave.open(p, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(data.tobytes())
_tmpfile("h3tl_tmp.wav", _mk_wav)

img_media = _tmp[0] and {"name": "h3tl_tmp.png", "subfolder": "", "type": "input"}
vid_media = {"name": "h3tl_tmp.mp4", "subfolder": "", "type": "input"}
wav_media = {"name": "h3tl_tmp.wav", "subfolder": "", "type": "input"}

# image file clip: no input required
state = json.dumps({"clips": [{"id": 1, "kind": "image", "start": 4,
                               "strength": 1, "file": img_media}]})
cond = run(state, AudioVAE(), vae=FakeVideoVAE())
assert len(cond["minimax_keyframes"]) == 1
assert cond["minimax_keyframes"][0][pl.MC_KEY] == 3
print("uploaded image clip OK")

# video file clip with src_start: 16 frames, start at frame 10, 5 frames
# from source frame 10 -> window [10..15), run 5 -> 2 latent steps
avv = AudioVAE()
state = json.dumps({"clips": [{"id": 1, "kind": "video", "start": 10,
                               "len": 5, "src_start": 10,
                               "file": vid_media}]})
cond = run(state, avv, vae=FakeVideoVAE())
kfs = cond["minimax_keyframes"]
assert len(kfs) == 2, len(kfs)
assert kfs[0][pl.MC_KEY] == 9 and kfs[1][pl.MC_KEY] == 10
assert (cond.get("minimax_refs") or []) == []  # mp4 has no audio -> silent
print("uploaded video file clip OK: src_start window, silent track")

# video file WITH an audio track: the file's own sound becomes a ref with
# no audio input connected
import subprocess as _sp
_av_raw = os.path.join(inp_dir, "h3tl_tmp_av.raw.mp4")
def _mk_video_with_audio(p):
    _mk_video(_av_raw)
    _sp.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", _av_raw,
             "-i", os.path.join(inp_dir, "h3tl_tmp.wav"),
             "-c:v", "copy", "-c:a", "aac", p],
            check=True, capture_output=True)
avmp4 = _tmpfile("h3tl_tmp_av.mp4", _mk_video_with_audio)
_tmp.append(_av_raw)
ava2 = AudioVAE()
state = json.dumps({"clips": [{"id": 1, "kind": "video", "start": 10,
                               "len": 5, "file": avmp4}]})
cond = run(state, ava2, vae=FakeVideoVAE())
refs = cond["minimax_refs"]
assert len(refs) == 1, refs
assert refs[0][pl.MC_AUDIO_KEY] == 14.0, refs[0][pl.MC_AUDIO_KEY]
print("uploaded video-with-audio file clip OK: file sound becomes a ref")

# separated band: unlinked file video keeps its own audio window
ava3 = AudioVAE()
state = json.dumps({"clips": [{"id": 1, "kind": "video", "start": 10,
                               "len": 5, "audio_link": False,
                               "audio_start": 40, "audio_len": 5,
                               "file": avmp4}]})
cond = run(state, ava3, vae=FakeVideoVAE())
refs = cond["minimax_refs"]
assert len(refs) == 1, refs
assert refs[0][pl.MC_AUDIO_KEY] == 44.0, refs[0][pl.MC_AUDIO_KEY]
print("unlinked file band OK: separated audio injects at its own position")

# audio_off also suppresses the file's own sound
avo2 = AudioVAE()
state = json.dumps({"clips": [{"id": 1, "kind": "video", "start": 4,
                               "len": 5, "audio_off": True,
                               "file": vid_media}]})
cond = run(state, avo2, vae=FakeVideoVAE())
assert len(cond["minimax_keyframes"]) == 2
assert (cond.get("minimax_refs") or []) == []
print("audio_off file video OK: file sound suppressed")

# audio file clip with src_start: 2 s wav, drop 1 s, window the rest
ava = AudioVAE()
state = json.dumps({"clips": [{"id": 1, "kind": "audio", "start": 30,
                               "len": 22, "src_start": 24,
                               "file": wav_media}]})
cond = run(state, ava, vae=FakeVideoVAE())
refs = cond["minimax_refs"]
assert len(refs) == 1
assert refs[0][pl.MC_AUDIO_KEY] == 51.0, refs[0][pl.MC_AUDIO_KEY]  # 30-1+22
w = ava.windows[0]
want = int(round(22 / 24.0 * 16000))
assert int(w.shape[1]) == min(want, 16000), w.shape  # 1 s left after slice
print("uploaded audio file clip OK: src_start slice + window")

# missing file raises
try:
    run(json.dumps({"clips": [{"id": 1, "kind": "image", "start": 4,
                               "file": {"name": "missing.png",
                                        "type": "input"}}]}),
        AudioVAE(), vae=FakeVideoVAE())
    raise AssertionError("missing media file must raise")
except ValueError:
    pass
print("missing media file validation OK")

for p in _tmp:
    try:
        os.remove(p)
    except OSError:
        pass


print("PASS: timeline super node")
