"""Static checks for the multi-ref timeline-audio compatibility fork.

Runs without ComfyUI/GPU. The real PackedLayout numerical self-test is also
embedded in patch_layout.py and runs when the custom node is imported by ComfyUI.
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

    assert 'append=True' in NODES
    assert '{"minimax_refs": [ref]}, append=True)' in NODES
    assert 'values["minimax_refs"] = [ref]' not in NODES

    # Multi-ref timeline audio: marked blocks may appear anywhere among the
    # refs and are selected from a pre-shift snapshot, one shift per block.
    assert 'marked = [(i, r) for i, r in enumerate(refs)' in LAYOUT
    assert 'snapshot = t.clone()' in LAYOUT
    assert 'for i, blk in marked:' in LAYOUT
    assert '_ref_cursor_advance(refs[:i])' in LAYOUT
    assert 'H3_MC_MULTI_REF_AUDIO_SHAREABLE_SELFTEST' in LAYOUT
    assert 'multi-mark audio' in LAYOUT

    # Per-block audio strength: each block overrides the global
    # audio_cond_noise_aug from its own ref dict.
    assert 'MC_AUDIO_STRENGTH = "motion_context_audio_strength"' in LAYOUT
    assert '_patched_cond_audio_rows' in PAYLOAD
    assert 'apply_cond_audio_patch' in PAYLOAD

    # Duplicate-install hardening: candidate takeover names, sibling-folder
    # scan, and the self-test failure message telling the user to delete
    # other H3-Motion-Context copies.
    assert '_FOREIGN_ORIG_NAMES' in LAYOUT
    assert '_find_dup_installs' in LAYOUT
    assert '_original_init' in LAYOUT
    assert 'delete ALL of them' in LAYOUT
    assert 'every copy EXCEPT this fork' in LAYOUT
    assert 'DELETE every other' in NODES

    assert 'class MiniMaxH3CustomAudio' in NODES
    assert '"MiniMaxH3CustomAudio": MiniMaxH3CustomAudio,' in NODES
    assert '_encode_audio_window' in NODES
    assert 'MC_AUDIO_STRENGTH: strength' in NODES

    # Continuous strength: the block is never dropped; every denoising step
    # blends the clip with the model's own generation at the block's steps.
    assert '_audio_blend_map' in PAYLOAD
    assert '_h3mc_audio_blend_rows' in PAYLOAD
    assert 'apply_forward_patch' in PAYLOAD
    assert 's * r + (1.0 - s) * target' in PAYLOAD
    assert '_apply_forward_patch' in NODES

    print("PASS: multi-ref timeline-audio fork structure + Python syntax")


if __name__ == "__main__":
    main()
