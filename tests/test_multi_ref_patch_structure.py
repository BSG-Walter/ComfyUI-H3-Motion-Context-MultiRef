"""Static checks for the multi-ref timeline-audio compatibility fork.

Runs without ComfyUI/GPU. The real PackedLayout numerical self-test is also
embedded in patch_layout.py and runs when the custom node is imported by ComfyUI.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NODES = (ROOT / "nodes.py").read_text(encoding="utf-8")
LAYOUT = (ROOT / "patch_layout.py").read_text(encoding="utf-8")


def main():
    compile(NODES, "nodes.py", "exec")
    compile(LAYOUT, "patch_layout.py", "exec")

    assert 'motion_context_audio_ref = ref' in NODES
    assert 'append=True' in NODES
    assert 'values["minimax_refs"] = [ref]' not in NODES

    assert 'marked_idx = [i for i, r in enumerate(refs)' in LAYOUT
    assert 'idx != len(refs) - 1' in LAYOUT
    assert '_ref_cursor_advance(refs[:idx])' in LAYOUT
    assert 'H3_MC_MULTI_REF_AUDIO_SHAREABLE_SELFTEST' in LAYOUT

    print("PASS: multi-ref timeline-audio fork structure + Python syntax")


if __name__ == "__main__":
    main()
