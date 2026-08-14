"""Runtime checks for the MiniMax H3 Timeline node using ComfyUI core guides."""

import importlib.util
import sys
from pathlib import Path
import torch

COMFY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMFY_ROOT))

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
        '{"id":1,"kind":"image","start":1},' \
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


# 3. Clamping and bounds check
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
assert len(kfs3) == 3
# Total frames is 85.
# Image clamped to frame_count - 1 = 84
assert kfs3[2]["resolved_frame_index"] == 84
# Video of 22 frames clamped to 85 - 22 = 63
assert kfs3[0]["resolved_frame_index"] == 63
# Audio clamped to 84
assert kfs3[1]["resolved_frame_index"] == 84

print("Test 3 OK: bounds clamping")


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

print("ALL TESTS PASSED!")
