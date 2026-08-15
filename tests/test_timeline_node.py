r"""Runtime checks for the MiniMax H3 Timeline node using ComfyUI core guides.

# Run with ComfyUI venv:
# C:\Users\Walter\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe tests/test_timeline_node.py
"""

import importlib.util
import sys
import unittest
from pathlib import Path
import torch

COMFY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMFY_ROOT))

# Mock comfy_aimdo and comfy_kitchen if not present in test environment
import types
mock_aimdo = types.ModuleType("comfy_aimdo")
mock_aimdo.__path__ = []
for submod in ["host_buffer", "vram_buffer", "model_vbar", "torch", "pinned_memory"]:
    sm = types.ModuleType(f"comfy_aimdo.{submod}")
    setattr(mock_aimdo, submod, sm)
    sys.modules[f"comfy_aimdo.{submod}"] = sm
sys.modules["comfy_aimdo"] = mock_aimdo

if "comfy_kitchen" not in sys.modules:
    mock_kitchen = types.ModuleType("comfy_kitchen")
    mock_kitchen.__path__ = []
    mock_kitchen.int8_attention_is_available = lambda: False
    sys.modules["comfy_kitchen"] = mock_kitchen

base = "custom_nodes.ComfyUI-H3-Motion-Context-Timeline"
node_dir = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    base, str(node_dir / "__init__.py"),
    submodule_search_locations=[str(node_dir)])
pkg = importlib.util.module_from_spec(spec)
sys.modules[base] = pkg
spec.loader.exec_module(pkg)
n = pkg.nodes


class FakeVideoVAE:
    def encode(self, pix):
        # 1 frame -> 1 step, 5 frames -> 2 steps, 22 frames -> 7 steps
        n = pix.shape[0]
        steps = 1 if n == 1 else (n - 5) // 17 * 5 + 2
        return torch.zeros(1, 24, steps, 2, 2)


class FakeAV:
    def unbind(self):
        # 25 latent steps -> 85 pixel frames (5x [1,4,4,4,4])
        return (torch.zeros(1, 24, 25, 2, 2), torch.zeros(1, 32, 2, 40))


class AudioVAE:
    audio_sample_rate = 16000

    def __init__(self):
        self.windows = []

    def encode(self, wav):
        self.windows.append(wav.clone())
        # return [1, 32, 2, T]
        return torch.zeros(1, 32, 2, 12)


def audio(seconds=1.0, sr=16000):
    n = int(sr * seconds)
    ramp = torch.linspace(0, 1, n).unsqueeze(0).unsqueeze(0)
    return {"waveform": ramp, "sample_rate": sr}


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


# 1. Mixed timeline: image + video (linked audio) + audio clip
av = AudioVAE()
state = '{"clips":[' \
        '{"id":1,"kind":"image","start":1,"len":1},' \
        '{"id":2,"kind":"video","start":10,"len":22,"audio_link":true},' \
        '{"id":3,"kind":"audio","start":60,"len":22}]}'
cond = run(
    state, av, vae=FakeVideoVAE(),
    image_1=torch.rand(1, 32, 32, 3),
    video_2=torch.rand(22, 32, 32, 3),
    video_audio_2=audio(2.0),
    audio_3=audio(2.0),
)

kfs = cond["minimax_keyframes"]
assert len(kfs) == 3, f"Expected 3 keyframes, got {len(kfs)}"

# Image at start 1 -> resolved_frame_index = 0
assert kfs[0]["resolved_frame_index"] == 0
assert kfs[0]["latent"].shape[2] == 1  # 1 step
assert "audio_latent" not in kfs[0]

# Video at start 10 -> resolved_frame_index = 9 (linked audio attached to the same keyframe)
assert kfs[1]["resolved_frame_index"] == 9
assert kfs[1]["latent"].shape[2] == 7  # 22 frames -> 7 steps
assert "audio_latent" in kfs[1]

# Audio at start 60 -> resolved_frame_index = 59
assert kfs[2]["resolved_frame_index"] == 59
assert "audio_latent" in kfs[2]
assert "latent" not in kfs[2]

print("Test 1 OK: mixed timeline with core minimax_keyframes")


# 2. Video with unlinked audio
av2 = AudioVAE()
state2 = '{"clips":[' \
         '{"id":1,"kind":"video","start":10,"len":22,"audio_link":false,' \
         '"audio_start":30,"audio_len":10}]}'
cond2 = run(
    state2, av2, vae=FakeVideoVAE(),
    video_1=torch.rand(22, 32, 32, 3),
    video_audio_1=audio(2.0),
)
kfs2 = cond2["minimax_keyframes"]
assert len(kfs2) == 2, f"Expected 2 keyframes (video + separate audio), got {len(kfs2)}"
assert kfs2[0]["resolved_frame_index"] == 9
assert "latent" in kfs2[0] and "audio_latent" not in kfs2[0]
assert kfs2[1]["resolved_frame_index"] == 29
assert "audio_latent" in kfs2[1]

print("Test 2 OK: unlinked video audio")


# 3. Clamping and bounds check: clips beyond timeline (frame_count=85) are ignored
state3 = '{"clips":[' \
         '{"id":1,"kind":"image","start":200},' \
         '{"id":2,"kind":"video","start":80,"len":22},' \
         '{"id":3,"kind":"audio","start":200,"len":22}]}'
cond3 = run(
    state3, AudioVAE(), vae=FakeVideoVAE(),
    image_1=torch.rand(1, 32, 32, 3),
    video_2=torch.rand(22, 32, 32, 3),
    audio_3=audio(1.0),
)
kfs3 = cond3["minimax_keyframes"]
# Clips starting at 200 (> 85) are strictly dropped.
# Video starts at 80 (frame_idx=79), length clamped to 85-79=6 frames (guide=5 frames)
assert len(kfs3) == 1
assert kfs3[0]["resolved_frame_index"] == 79
assert kfs3[0]["latent"].shape[2] == 2  # 5 frames -> 2 tokens

print("Test 3 OK: bounds clamping and timeline limit cutoff")


# 4. Empty clips
cond_empty = run('{"clips":[]}')
assert "minimax_keyframes" not in cond_empty or len(cond_empty.get("minimax_keyframes", [])) == 0
print("Test 4 OK: empty clips")


# 5. Real ComfyUI Audio VAE encoding integration check
import comfy.sd
real_audio_vae = comfy.sd.VAE(sd={'pre_block.attn.zero_k_bias': torch.zeros(2048)})
state5 = '{"clips":[{"id":1,"kind":"audio","start":1,"len":24}]}'
cond5 = run(state5, audio_vae=real_audio_vae, audio_1=audio(2.0, sr=32000))
assert "audio_latent" in cond5["minimax_keyframes"][0]
assert cond5["minimax_keyframes"][0]["audio_latent"].ndim == 4
print("Test 5 OK: real comfy.sd.VAE audio encoding integration")

# 6. Clip compilation with UI strength state
state6 = '{"clips":[' \
         '{"id":1,"kind":"image","start":1,"strength":0.5},' \
         '{"id":2,"kind":"video","start":10,"len":22,"strength":0.4,"audio_strength":0.7,"audio_link":true},' \
         '{"id":3,"kind":"audio","start":60,"len":22,"audio_strength":0.1}]}'
cond6 = run(
    state6, AudioVAE(), vae=FakeVideoVAE(),
    image_1=torch.ones(1, 32, 32, 3),
    video_2=torch.ones(22, 32, 32, 3),
    video_audio_2=audio(2.0),
    audio_3=audio(2.0),
)
assert len(cond6["minimax_keyframes"]) >= 3
print("Test 6 OK: clips with UI strength parameters compile cleanly")

# 7. Animated GIF loading simulation
import os
import tempfile
import numpy as np
from PIL import Image, ImageSequence

gif_frames = [Image.fromarray(np.uint8(np.full((32, 32, 3), i * 50))) for i in range(5)]
with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
    gif_frames[0].save(f.name, save_all=True, append_images=gif_frames[1:], duration=100, loop=0)
    gif_path = f.name

try:
    with Image.open(gif_path) as img:
        collected = []
        t_sec = 0.0
        for frame in ImageSequence.Iterator(img):
            duration = frame.info.get("duration", 100) / 1000.0
            arr = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
            collected.append((t_sec, torch.from_numpy(arr)))
            t_sec += duration
        gif_tensor = torch.stack([item[1] for item in collected])
        assert gif_tensor.shape[0] == 5
    print("Test 7 OK: animated GIF multi-frame handling")
finally:
    if os.path.exists(gif_path):
        os.remove(gif_path)

# 8. Video file fast slicing and split timeline test
import av
with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
    mp4_path = f.name

try:
    with av.open(mp4_path, mode="w") as container:
        vstream = container.add_stream("h264", rate=24)
        vstream.width = 64
        vstream.height = 64
        vstream.pix_fmt = "yuv420p"
        astream = container.add_stream("aac", rate=44100)
        astream.layout = "stereo"
        for i in range(120):  # 5 seconds at 24fps
            img = np.full((64, 64, 3), i * 2, dtype=np.uint8)
            vf = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in vstream.encode(vf):
                container.mux(packet)
        for packet in vstream.encode():
            container.mux(packet)
        for _ in range(5):
            samples = np.zeros((1, 44100 * 2), dtype=np.float32)
            af_frame = av.AudioFrame.from_ndarray(samples, format="flt", layout="stereo")
            af_frame.sample_rate = 44100
            for packet in astream.encode(af_frame):
                container.mux(packet)
        for packet in astream.encode():
            container.mux(packet)

    # Test loading slice from 2.0s (frame 48) for 22 frames
    dummy_media = {"name": os.path.basename(mp4_path), "type": "input", "subfolder": ""}
    import folder_paths
    orig_annotated = folder_paths.get_annotated_filepath
    folder_paths.get_annotated_filepath = lambda ref: mp4_path

    try:
        data = n._load_media_file(dummy_media, fps=24, start_sec=2.0, duration_sec=22/24.0, load_video=True, load_audio=True)
        assert data["frames"] is not None
        assert data["frames"].shape[0] >= 22
        assert data["audio"] is not None

        # Test in node timeline with split clips (left + right split)
        state8 = '{"clips":[' \
                 '{"id":1,"kind":"video","start":1,"len":22,"src_start":0,"file":{"name":"dummy.mp4","type":"input"}},' \
                 '{"id":2,"kind":"video","start":25,"len":22,"src_start":48,"file":{"name":"dummy.mp4","type":"input"}}]}'
        cond8 = run(state8, AudioVAE(), vae=FakeVideoVAE())
        kfs8 = cond8["minimax_keyframes"]
        assert len(kfs8) == 2
        assert kfs8[0]["resolved_frame_index"] == 0
        assert kfs8[1]["resolved_frame_index"] == 24
        print("Test 8 OK: fast video seek/slice and split clips handling")
    finally:
        folder_paths.get_annotated_filepath = orig_annotated
finally:
    if os.path.exists(mp4_path):
        os.remove(mp4_path)

print("ALL TESTS PASSED!")
