"""Runtime checks for the Custom Video node (multi-slot, per-clip strength).

Needs ComfyUI's own python (torch + comfy.ldm.minimax importable), i.e. the
.venv next to ComfyUI, not the bare python on PATH. No GPU required.

    & ComfyUI/.venv/Scripts/python.exe tests/test_custom_video_runtime.py

The layout patch self-test runs inside apply_patch(); the rest here checks
the node-level wiring: dynamic input map, multi-clip apply with per-clip
strength and audio, and the video/audio pin-then-flip schedule.
"""

import importlib.util
import json
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
n, pl, pp = pkg.nodes, pkg.patch_layout, pkg.patch_payload


# --- dynamic input map: legacy, multi-prefix, fixed keys, serialization ----
legacy = n._DynamicInputs("keyframe_image_", ("IMAGE",))
assert "keyframe_image_9" in legacy and legacy["keyframe_image_9"] == ("IMAGE",)

opt = n._DynamicInputs(
    ("video_", ("IMAGE",)), ("video_audio_", ("AUDIO",)),
    audio_vae=("VAE", {"tooltip": "x"}))
assert "video_2" in opt and opt["video_2"] == ("IMAGE",)
assert "video_audio_2" in opt and opt["video_audio_2"] == ("AUDIO",)
assert "audio_vae" in opt and opt["audio_vae"][0] == "VAE"
assert opt.get("nope") is None
assert '"audio_vae"' in json.dumps({"optional": opt})

it = n.MiniMaxH3CustomVideo.INPUT_TYPES()
assert it["optional"]["video_7"] == ("IMAGE",)
assert it["optional"]["video_audio_7"] == ("AUDIO",)
assert it["optional"]["audio_vae"][0] == "VAE"
print("dynamic input map OK: legacy + multi-prefix + fixed-key serialization")


# --- node apply: 2 clips, per-clip strength, per-clip audio, audio follows
# the video strength slider ---
class FakeVAE:
    def encode(self, pix):
        return torch.zeros(pix.shape[0], 4, 2, 2, 2)


class FakeAV:
    def unbind(self):
        return (torch.zeros(1, 4, 7, 1, 1), torch.zeros(1, 4, 2, 16))


node = n.MiniMaxH3CustomVideo()
video1 = torch.rand(5, 12, 12, 3)
video2 = torch.rand(5, 12, 12, 3)
audio = {"waveform": torch.rand(1, 1, 16000), "sample_rate": 16000}
state = '{"count":2,"positions":[1,10],"strengths":[0.5,1.0]}'
out = node.apply([[torch.zeros(1, 7, 4), {}]], FakeVAE(),
                 {"samples": FakeAV()}, state, "1-based", "disabled",
                 audio_vae=FakeVAE(),
                 video_1=video1, video_2=video2, video_audio_1=audio)
cond = out[0][0][1]
kfs = cond["minimax_keyframes"]
assert len(kfs) == 4, len(kfs)
assert cond["minimax_frame_count"] == 22
assert kfs[0][pl.MC_KEY] == 0 and kfs[0][pl.MC_VIDEO_STRENGTH] == 0.5
assert kfs[1][pl.MC_KEY] == 1 and kfs[1][pl.MC_VIDEO_STRENGTH] == 0.5
assert kfs[2][pl.MC_KEY] == 9 and kfs[2][pl.MC_VIDEO_STRENGTH] == 1.0
assert kfs[3][pl.MC_KEY] == 10 and kfs[3][pl.MC_VIDEO_STRENGTH] == 1.0
refs = cond["minimax_refs"]
assert len(refs) == 1 and refs[0][pl.MC_AUDIO_KEY] == 5.0
assert refs[0][pl.MC_AUDIO_STRENGTH] == 0.5
print("video node apply OK: 2 clips, per-clip strength, per-clip audio")


# --- video pin/flip: marked rows are pinned CLEAN (never noised); the
# forward wrapper claims them clean (0.999), records the active set, and
# flips weak blocks out of the layout once t_v crosses their strength. The
# payload lists are NEVER mutated: the packers read latents off the dicts
# and skip inactive blocks, so rows always match the layout by construction.
assert pl.apply_patch(), "layout patch must apply"
mm = pl.mm
m = mm.MiniMaxH3Model.__new__(mm.MiniMaxH3Model)
m.patch_size = (1, 2, 2)
pp._ORIG["_cond_video_rows"] = pp._patched_cond_video_rows._h3mc_orig
z = torch.randn(1, 24, 1, 2, 2)
z2 = torch.randn(1, 24, 1, 2, 2)
clip_rows = mm.patchify_video(z.to(torch.float32), m.patch_size)
clip2_rows = mm.patchify_video(z2.to(torch.float32), m.patch_size)
kf_weak = {"resolved_frame_index": 0, pl.MC_KEY: 0, pl.MC_VIDEO_STRENGTH: 0.5,
           "latent": z}
kf_strong = {"resolved_frame_index": 0, pl.MC_KEY: 1, pl.MC_VIDEO_STRENGTH: 1.0,
             "latent": z2}
video_payload = {"keyframes": [kf_weak, kf_strong], "seed": 0,
                 "_h3mc_active_keyframes": [kf_weak, kf_strong]}
rows = pp._patched_cond_video_rows(m, video_payload, torch.device("cpu"))
assert torch.equal(rows, torch.cat([clip_rows, clip2_rows]))  # both clean
# flipped weak block: dropped from the active set -> its rows disappear
video_payload["_h3mc_active_keyframes"] = [kf_strong]
rows = pp._patched_cond_video_rows(m, video_payload, torch.device("cpu"))
assert torch.equal(rows, clip2_rows)
# unmarked runs take the stock packer wholesale (active set absent)
unmarked = {"keyframes": [kf_strong], "cond_video_latents": [z2], "seed": 0,
            "visual_cond_noise_aug": 1.0}
assert torch.equal(pp._patched_cond_video_rows(m, unmarked,
                                               torch.device("cpu")),
                   clip2_rows)

saved_forward = pp._ORIG.get("forward")
calls = []
za = torch.randn(1, 32, 2, 6)
def rec_forward(self, x, timestep, context, transformer_options={},
                minimax_payload=None, **kwargs):
    calls.append(float(timestep.flatten()[0]))
    return x
pp._ORIG["forward"] = rec_forward
try:
    x = (torch.zeros(1, 24, 8, 4, 4), torch.zeros(1, 32, 16))
    ctx = torch.zeros(1, 5, 8)
    topts = {"minimax_h3_sigma_shift_video": 12.0,
             "minimax_h3_sigma_shift_audio": 3.0}
    frame_count = sum(mm.FRAME_PER_TOKEN[k % 5] for k in range(8))
    ref_weak_a = {"kind": "audio", "ref_audio_t": 6, pl.MC_AUDIO_KEY: 9.0,
                  pl.MC_AUDIO_STRENGTH: 0.6, "audio_latent": za}
    payload = {"keyframes": [kf_weak, kf_strong], "refs": [ref_weak_a],
               "seed": 0, "frame_count": frame_count, "layout": "SENTINEL"}
    pp._patched_forward(m, x, torch.tensor([800.0]), ctx, topts, payload)
    assert payload["visual_cond_noise_aug"] == 0.999
    assert payload["audio_cond_noise_aug"] == 0.999
    assert len(payload["keyframes"]) == 2  # t_v 0.2 < 0.5: no video flip
    assert len(payload["refs"]) == 1       # t_a 0.5 < 0.6: no audio flip
    assert payload["layout"] == "SENTINEL"  # set unchanged: no rebuild
    pp._patched_forward(m, x, torch.tensor([200.0]), ctx, topts, payload)
    # lists never mutated; the flip is recorded in the active sets
    assert [kf[pl.MC_KEY] for kf in payload["keyframes"]] == [0, 1]
    assert payload["refs"] == [ref_weak_a]
    assert payload["_h3mc_active_keyframes"] == [kf_strong]
    assert payload["_h3mc_active_refs"] == []
    assert payload["layout"] != "SENTINEL"  # rebuilt once on the flip
    rebuilt = payload["layout"]
    segs = [k for _, _, k in rebuilt.segments]
    assert sum(1 for k in segs if k == "cond") == 1, segs
    assert sum(1 for k in segs if k == "ref_audio") == 0, segs
    # same active set again: layout object is not rebuilt again
    pp._patched_forward(m, x, torch.tensor([200.0]), ctx, topts, payload)
    assert payload["layout"] is rebuilt
    # the patched packer skips flipped blocks -> rows match the layout
    vrows = pp._patched_cond_video_rows(m, payload, torch.device("cpu"))
    assert torch.equal(vrows, clip2_rows)
    arows = pp._patched_cond_audio_rows(m, payload, torch.device("cpu"))
    assert arows is None
    assert len(calls) == 3
    # unmarked runs pass straight through, payload untouched
    plain = {"keyframes": [kf_strong], "cond_video_latents": [z2],
             "seed": 0, "layout": "SENTINEL2"}
    pp._patched_forward(m, x, torch.tensor([200.0]), ctx, topts, plain)
    assert plain["layout"] == "SENTINEL2" and "visual_cond_noise_aug" not in plain
    assert len(calls) == 4
finally:
    pp._ORIG["forward"] = saved_forward
print("video pin/flip OK: clean pin below t_v, layout flip above")

za = torch.randn(1, 32, 2, 6)
za2 = torch.randn(1, 32, 2, 6)
packed = mm.pack_audio(za.to(torch.float32))
packed2 = mm.pack_audio(za2.to(torch.float32))
ref_weak = {"kind": "audio", "ref_audio_t": 6, pl.MC_AUDIO_KEY: 9.0,
            pl.MC_AUDIO_STRENGTH: 0.5, "audio_latent": za}
ref_strong = {"kind": "audio", "ref_audio_t": 6, pl.MC_AUDIO_KEY: 12.0,
              pl.MC_AUDIO_STRENGTH: 1.0, "audio_latent": za2}
audio_payload = {"refs": [ref_weak, ref_strong],
                 "cond_audio_latents": [za, za2], "seed": 0}
rows = pp._patched_cond_audio_rows(m, audio_payload, torch.device("cpu"))
assert torch.equal(rows, torch.cat([packed, packed2]))  # both pinned clean
# unmarked refs with global aug 1.0 keep stock behaviour: clean rows
unmarked = {"refs": [dict(ref_weak, **{pl.MC_AUDIO_STRENGTH: None})],
            "cond_audio_latents": [za], "seed": 0, "audio_cond_noise_aug": 1.0}
assert torch.equal(pp._patched_cond_audio_rows(m, unmarked,
                                               torch.device("cpu")),
                   packed)
print("audio pin OK: marked rows always clean, unmarked stock")