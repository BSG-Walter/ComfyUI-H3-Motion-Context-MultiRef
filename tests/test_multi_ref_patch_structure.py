"""Static checks for the multi-ref timeline-audio compatibility fork.

Runs without ComfyUI/GPU. The PackedLayout numerical self-test is embedded
in patch_layout.py and runs at import, so the layout formula is not pinned
here; the payload patches (keyframe/ref coexistence, per-block audio
strength) have no numeric self-test, so their structure is.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NODES = (ROOT / "nodes.py").read_text(encoding="utf-8")
LAYOUT = (ROOT / "patch_layout.py").read_text(encoding="utf-8")
PAYLOAD = (ROOT / "patch_payload.py").read_text(encoding="utf-8")


def main():
    compile(NODES, "nodes.py", "exec")
    compile(LAYOUT, "patch_layout.py", "exec")
    compile(PAYLOAD, "patch_payload.py", "exec")

    # Coexistence: the MC audio ref appends to existing Ref2VA refs.
    assert '{"minimax_refs": [ref]}, append=True)' in NODES
    assert 'values["minimax_refs"] = [ref]' not in NODES

    # Per-block audio strength: MC_AUDIO_STRENGTH rides on each ref and the
    # patched row packer blends weak blocks with the model's own audio.
    assert 'MC_AUDIO_STRENGTH: strength' in NODES
    assert '_patched_cond_audio_rows' in PAYLOAD
    assert 's * r + (1.0 - s) * target' in PAYLOAD
    assert '_audio_blend_map' in PAYLOAD
    assert 'apply_forward_patch' in PAYLOAD

    # Duplicate-install hardening lives in the layout patcher.
    assert '_FOREIGN_ORIG_NAMES' in LAYOUT
    assert '_find_dup_installs' in LAYOUT

    print("PASS: multi-ref timeline-audio fork structure + Python syntax")


if __name__ == "__main__":
    main()
