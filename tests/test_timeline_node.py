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
        # real H3 video VAE: 1 frame -> 1 step, else ceil(n/17)*5-3
        n = pix.shape[0]
        steps = 1 if n == 1 else (n + 16) // 17 * 5 - 3
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


# --- the video's audio band has its own strength and envelope ---
avb = AudioVAE()
state = '{"clips":[{"id":1,"kind":"video","start":10,"len":22,' \
        '"audio_link":true,"strength":0.9,' \
        '"audio_strength":0.3,"audio_env":[[0,0.2],[10,1.0]]}]}'
cond = run(state, avb, vae=FakeVideoVAE(),
           video_1=torch.rand(5, 12, 12, 3), video_audio_1=audio())
assert all(kf[pl.MC_VIDEO_STRENGTH] == 0.9
           for kf in cond["minimax_keyframes"])  # video untouched
refs = cond["minimax_refs"]
assert len(refs) == 12, len(refs)  # band env has points -> per-step refs
assert all(r["ref_audio_t"] == 1 for r in refs)
assert refs[0][pl.MC_AUDIO_STRENGTH] == 0.2, \
    refs[0][pl.MC_AUDIO_STRENGTH]
assert abs(refs[11][pl.MC_AUDIO_STRENGTH] - 17 / 30.0) < 1e-9
print("video audio band OK: own strength + envelope, video untouched")


# --- strength 0 is allowed: prompt-only from the start ---
cond = run('{"clips":[{"id":1,"kind":"image","start":1,' \
           '"len":22,"strength":0,"env":[[0,0]]}]}',
           AudioVAE(), vae=FakeVideoVAE(),
           image_1=torch.rand(1, 12, 12, 3))
kfs = cond["minimax_keyframes"]
assert len(kfs) == 7
assert all(kf[pl.MC_VIDEO_STRENGTH] == 0.0 for kf in kfs)
cond = run('{"clips":[{"id":1,"kind":"audio","start":1,"len":5,' \
           '"strength":0}]}', AudioVAE(), vae=FakeVideoVAE(),
           audio_1=audio())
assert cond["minimax_refs"][0][pl.MC_AUDIO_STRENGTH] == 0.0
print("zero strength OK: flat and envelope at 0")


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


# --- strength envelope: per-frame video strengths + per-step audio ---
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
# both audio windows have varying envelopes, so every ref is one latent
# step with its own linearly interpolated strength (like video keyframes)
assert len(refs) == 24, len(refs)
assert all(r["ref_audio_t"] == 1 for r in refs), \
    [r["ref_audio_t"] for r in refs]
va, seg = refs[:12], refs[12:]
assert va[0][pl.MC_AUDIO_STRENGTH] == 0.2
assert abs(va[11][pl.MC_AUDIO_STRENGTH] - 17 / 30.0) < 1e-9
assert va[11][pl.MC_AUDIO_KEY] == 14.0, va[11][pl.MC_AUDIO_KEY]
assert len([r[pl.MC_AUDIO_STRENGTH] for r in seg][:6]) == 6 and \
    all(abs(a - b) < 1e-9 for a, b in
        zip([r[pl.MC_AUDIO_STRENGTH] for r in seg][:6],
            [1.0, 11 / 12.0, 5 / 6.0, 3 / 4.0, 2 / 3.0, 7 / 12.0])), \
    [r[pl.MC_AUDIO_STRENGTH] for r in seg][:6]
assert abs(seg[6][pl.MC_AUDIO_STRENGTH] - 0.5) < 1e-9  # exactly on the point
assert abs(seg[11][pl.MC_AUDIO_STRENGTH] - 0.225) < 1e-9
assert abs(seg[0][pl.MC_AUDIO_KEY] - (59 + 22 / 12.0)) < 1e-9
assert seg[-1][pl.MC_AUDIO_KEY] == 81.0, seg[-1][pl.MC_AUDIO_KEY]
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
print("strength envelope OK: per-frame video strengths + per-step audio")


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


# --- stretched image: still broadcast + per-step envelope strengths ---
state = '{"clips":[{"id":1,"kind":"image","start":10,"len":22,' \
        '"env":[[0,0.2],[10,1.0]],"strength":0.7}]}'
cond = run(state, AudioVAE(), vae=FakeVideoVAE(),
           image_1=torch.rand(1, 12, 12, 3))
kfs = cond["minimax_keyframes"]
assert len(kfs) == 7, len(kfs)  # 22 frames -> 7 latent steps (1,4,4,4,4,1,4)
assert [kf[pl.MC_KEY] for kf in kfs] == [9, 10, 14, 18, 22, 26, 27], \
    [kf[pl.MC_KEY] for kf in kfs]
ss = [kf[pl.MC_VIDEO_STRENGTH] for kf in kfs]
assert ss[0] == 0.2 and abs(ss[1] - 0.28) < 1e-9 and ss[-1] == 1.0, ss
print("stretched image OK: broadcast still + envelope strengths")


# --- off-grid stretch covers a bit more; out-of-range parks the hold ---
cond = run('{"clips":[{"id":1,"kind":"image","start":1,"len":6}]}',
           AudioVAE(), vae=FakeVideoVAE(),
           image_1=torch.rand(1, 12, 12, 3))
kfs = cond["minimax_keyframes"]
assert len(kfs) == 3, len(kfs)  # 6 frames -> 3 steps covering 9
assert [kf[pl.MC_KEY] for kf in kfs] == [0, 1, 5]
cond = run('{"clips":[{"id":1,"kind":"image","start":80,"len":22}]}',
           AudioVAE(), vae=FakeVideoVAE(),
           image_1=torch.rand(1, 12, 12, 3))
kfs = cond["minimax_keyframes"]
assert len(kfs) == 7, len(kfs)
assert kfs[0][pl.MC_KEY] == 63, kfs[0][pl.MC_KEY]  # parked: 85 - 22
cond = run('{"clips":[{"id":1,"kind":"image","start":1,"len":200}]}',
           AudioVAE(), vae=FakeVideoVAE(),
           image_1=torch.rand(1, 12, 12, 3))
kfs = cond["minimax_keyframes"]
assert len(kfs) == 59, len(kfs)  # 200 frames -> 59 steps
assert kfs[0][pl.MC_KEY] == 0  # holds the whole clip, parked at frame 1
print("off-grid/parked stretched image OK")


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

# src_start past the end of the source (trimmed beyond the file): clamps to
# the last frame instead of crashing on an empty slice
avv2 = AudioVAE()
state = json.dumps({"clips": [{"id": 1, "kind": "video", "start": 10,
                               "len": 5, "src_start": 99,
                               "file": vid_media}]})
cond = run(state, avv2, vae=FakeVideoVAE())
kfs = cond["minimax_keyframes"]
assert len(kfs) == 1, len(kfs)
assert kfs[0][pl.MC_KEY] == 9, kfs[0][pl.MC_KEY]
print("video file clip OK: src_start past the end holds the last frame")

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

# unlinked band with a frozen source window: the band slices the file at
# audio_src_start, never at the video's current src_start, so trimming the
# video no longer moves the band's sound
avb = AudioVAE()
state = json.dumps({"clips": [{"id": 1, "kind": "video", "start": 10,
                               "len": 5, "src_start": 32,
                               "audio_link": False,
                               "audio_start": 40, "audio_len": 5,
                               "audio_src_start": 24,
                               "file": avmp4}]})
cond = run(state, avb, vae=FakeVideoVAE())
refs = cond["minimax_refs"]
assert len(refs) == 1, refs
assert refs[0][pl.MC_AUDIO_KEY] == 44.0, refs[0][pl.MC_AUDIO_KEY]
# the band's window must come from the file sliced at audio_src_start (24
# frames = 1.0 s), never at the video's src_start (32 frames = 1.33 s)
raw = n._load_media_file(avmp4)["audio"]
want = int(round(5 / 24.0 * 16000))
expect = head(n._slice_audio(raw, 24 / 24.0)["waveform"], want)
wrong = head(n._slice_audio(raw, 32 / 24.0)["waveform"], want)
w = avb.windows[0]
assert torch.allclose(w[0], expect), "band not sliced at audio_src_start"
assert not torch.allclose(w[0], wrong), "band followed the video's src_start"
print("unlinked band OK: frozen audio_src_start slices its own file window")

# audio_off also suppresses the file's own sound
avo2 = AudioVAE()
state = json.dumps({"clips": [{"id": 1, "kind": "video", "start": 4,
                               "len": 5, "audio_off": True,
                               "file": vid_media}]})
cond = run(state, avo2, vae=FakeVideoVAE())
assert len(cond["minimax_keyframes"]) == 2
assert (cond.get("minimax_refs") or []) == []
print("audio_off file video OK: file sound suppressed")

# off-grid video lengths split into consecutive grid runs: 36 = 22+5+5+1+
# 1+1+1, every frame 0..35 covered, no silent tail-drop
avs = AudioVAE()
state = json.dumps({"clips": [{"id": 1, "kind": "video", "start": 1,
                               "len": 36, "audio_link": False}]})
cond = run(state, avs, vae=FakeVideoVAE(),
           video_1=torch.rand(36, 12, 12, 3))
kfs = cond["minimax_keyframes"]
assert len(kfs) == 15, len(kfs)  # 7 + 2 + 2 + 1 + 1 + 1 + 1 steps
assert sorted(kf[pl.MC_KEY] for kf in kfs) == \
    [0, 1, 5, 9, 13, 17, 18, 22, 23, 27, 28, 32, 33, 34, 35]
assert all(kf[pl.MC_VIDEO_STRENGTH] == 1.0 for kf in kfs)
# the same clip is hard-injected: the whole 36-frame window is chunk-aligned
# (36 = 17*2+2), so every token of it pins exactly, pixels 0..35, and the
# last token (span [35,39)) is kept even though it is mostly held-edge so
# the clip's final frame is sent
hard = cond.get("minimax_hard_video") or []
assert [h["index"] for h in hard] == list(range(12)), \
    [h["index"] for h in hard]
print("video grid split OK: 36 frames -> runs 22+5+5+1+1+1+1, "
      "full coverage")

# two contiguous video clips (1..19 and 20..36) are hard-injected as ONE
# block: a single chunk-aligned window encodes to 12 exact tokens covering
# both clips, with no held-edge seam at the clip boundary
avt = AudioVAE()
state = json.dumps({"clips": [
    {"id": 1, "kind": "video", "start": 1, "len": 19, "audio_link": False},
    {"id": 2, "kind": "video", "start": 20, "len": 17, "audio_link": False},
]})
cond = run(state, avt, vae=FakeVideoVAE(),
           video_1=torch.rand(19, 12, 12, 3),
           video_2=torch.rand(17, 12, 12, 3))
hard = cond.get("minimax_hard_video") or []
assert [h["index"] for h in hard] == list(range(12)), \
    [h["index"] for h in hard]
print("contiguous clips OK: 19+17 frames injected as one 12-token block")

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
