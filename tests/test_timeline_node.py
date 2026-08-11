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


# --- out-of-range audio clamps with a warning, never raises ---
av4 = AudioVAE()
state = '{"clips":[{"id":1,"kind":"audio","start":200,"len":22}]}'
cond = run(state, av4, vae=FakeVideoVAE(), audio_1=audio())
assert cond["minimax_refs"][0][pl.MC_AUDIO_KEY] == 85.0  # parked at last frame
print("out-of-range audio clamp OK")


# --- structural violations still raise ---
try:
    run('{"clips":[{"id":1,"kind":"video","start":200,"len":22}]}',
        AudioVAE(), vae=FakeVideoVAE(),
        video_1=torch.rand(5, 12, 12, 3))
    raise AssertionError("video that does not fit must raise")
except ValueError:
    pass
try:
    run('{"clips":[{"id":1,"kind":"image","start":90}]}',
        AudioVAE(), vae=FakeVideoVAE(),
        image_1=torch.rand(1, 12, 12, 3))
    raise AssertionError("image beyond the clip must raise")
except ValueError:
    pass
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


print("PASS: timeline super node")
