"""Repro of the step-9 shape-mismatch crash: node-like payload, real layout,
real patched packers, real stock _forward row assembly.

& ComfyUI/.venv/Scripts/python.exe tests/repro_flip_crash.py
"""
import importlib.util
import sys
from pathlib import Path

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
pp = pkg.patch_payload

import torch

import comfy.ldm.minimax.model as mm
from comfy.ldm.minimax.model import PackedLayout

MC_KEY = pkg.patch_layout.MC_KEY
MC_VIDEO_STRENGTH = pkg.patch_layout.MC_VIDEO_STRENGTH
MC_AUDIO_KEY = pkg.patch_layout.MC_AUDIO_KEY
MC_AUDIO_STRENGTH = pkg.patch_layout.MC_AUDIO_STRENGTH

assert mm.MiniMaxH3Model._forward is not None


def build_payload(n_v1=12, n_v2=7, s1=0.5, s2=1.0, audio_s=0.5,
                  ref_latent=False, frame_count=124, t=1):
    kfs = []
    for k in range(n_v1):
        kfs.append({"resolved_frame_index": 0, MC_KEY: 0 + k,
                    MC_VIDEO_STRENGTH: s1, "latent": torch.randn(1, 24, 1, 8, 8)})
    for k in range(n_v2):
        kfs.append({"resolved_frame_index": 0, MC_KEY: 99 + k,
                    MC_VIDEO_STRENGTH: s2, "latent": torch.randn(1, 24, 1, 8, 8)})
    refs = [{"kind": "audio", "ref_audio_t": 65, "audio_latent": torch.randn(1, 32, 2, 65),
             MC_AUDIO_KEY: 39.0, MC_AUDIO_STRENGTH: audio_s}]
    if ref_latent:
        # a realistic mixed ref (Ref2VA video_audio): audio AND video content
        refs[0]["kind"] = "video_audio"
        refs[0]["latent_t"] = 1
        refs[0]["latent_h"] = 8
        refs[0]["latent_w"] = 8
        refs[0]["latent"] = torch.randn(1, 24, 1, 8, 8)
    payload = {
        "keyframes": kfs,
        "refs": refs,
        "frame_count": frame_count,
        "cond_video_latents": [kf["latent"] for kf in kfs]
                              + ([r["latent"] for r in refs if "latent" in r] or []),
        "cond_audio_latents": [r["audio_latent"] for r in refs if r.get("audio_latent") is not None],
        "seed": 0,
        "layout": PackedLayout(5, t, 8, 8, 65, keyframes=kfs, refs=refs,
                               frame_count=frame_count),
    }
    return payload


def stock_row_assembly(m, payload, device):
    """Copy of stock _forward lines 566-586 (embed-pre). Returns (ok, msg)."""
    layout = payload["layout"]
    video_x = torch.zeros(1, 24, 8, 8, 8)   # dummy; only rows count matters
    video_rows = mm.patchify_video(video_x.to(torch.float32), m.patch_size)
    img_update = layout.img_update.to(device)
    cond_video_rows = pp._patched_cond_video_rows(m, payload, device)
    n_cond = int(img_update.shape[0]) - int(img_update.sum())
    if cond_video_rows is None:
        return True, "no cond rows"
    if cond_video_rows.shape[0] != n_cond:
        return False, ("value %s vs %d cond rows (img_update %s, %d kfs, "
                       "%d latents)"
                       % (tuple(cond_video_rows.shape), n_cond,
                          tuple(img_update.shape), len(payload["keyframes"]),
                          len(payload["cond_video_latents"])))
    return True, "ok"


m = mm.MiniMaxH3Model.__new__(mm.MiniMaxH3Model)
m.patch_size = (1, 2, 2)
m.sigma_shift_video = 12.0
m.sigma_shift_audio = 3.0
assert pp.apply_cond_audio_patch(), "audio packer patch must apply"
assert pp.apply_cond_video_patch(), "video packer patch must apply"
assert pp.apply_forward_patch(), "forward patch must apply"
device = torch.device("cpu")

sigmas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
topts = {"minimax_h3_sigma_shift_video": 12.0, "minimax_h3_sigma_shift_audio": 3.0}

cases = [
    ("v1 weak only (s2=1.0)", dict(s1=0.5, s2=1.0)),
    ("both weak", dict(s1=0.5, s2=0.5)),
    ("audio ref also carries latent", dict(s1=0.5, s2=1.0, ref_latent=True)),
    ("audio strong, v1 weak", dict(s1=0.5, s2=1.0, audio_s=1.0)),
]
for name, kw in cases:
    payload = build_payload(**kw)
    print("  payload: kfs=%d refs=%d cv=%d ca=%d" % (
        len(payload["keyframes"]), len(payload["refs"]),
        len(payload["cond_video_latents"]),
        len(payload.get("cond_audio_latents") or [])))
    bad = False
    for i, sig in enumerate(sigmas):
        t = torch.tensor([sig * 1000.0])
        x = (torch.zeros(1, 24, 8, 8, 8), torch.zeros(1, 32, 2, 65))
        try:
            pp._patched_forward(m, x, t, torch.zeros(1, 5, 8), topts, payload)
        except AttributeError:
            pass  # row assembly passed; _forward stopped at the fake model's proj layers
        except RuntimeError as e:
            print("CASE %s | step %d sigma %.2f -> RUNTIME ERROR: %s" % (name, i, sig, e))
            bad = True
            break
        ok, msg = stock_row_assembly(m, payload, device)
        if not ok:
            print("CASE %s | step %d sigma %.2f -> MISMATCH: %s" % (name, i, sig, msg))
            bad = True
            break
    print("%-40s %s" % (name, "CRASHED" if bad else "consistent"))
