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
TIMELINE_JS = (ROOT / "js" / "h3_timeline.js").read_text(encoding="utf-8")


def main():
    compile(NODES, "nodes.py", "exec")
    compile(LAYOUT, "patch_layout.py", "exec")
    compile(PAYLOAD, "patch_payload.py", "exec")

    # Coexistence: the MC audio ref appends to existing Ref2VA refs.
    assert '{"minimax_refs": [ref]}, append=True)' in NODES
    assert 'values["minimax_refs"] = [ref]' not in NODES

    # Per-block audio strength: MC_AUDIO_STRENGTH rides on each ref and the
    # pin-then-flip machinery (forward wrapper + clean-pin packer) governs
    # it exactly like video.
    assert 'MC_AUDIO_STRENGTH: strength' in NODES
    assert '_patched_cond_audio_rows' in PAYLOAD
    assert 'apply_cond_audio_patch' in PAYLOAD

    # Per-block video strength (Custom Video): MC_VIDEO_STRENGTH rides on
    # each keyframe, the forward wrapper claims pinned rows clean (0.999)
    # and flips weak blocks out of the layout once their timeline crosses
    # their strength, and the patched packers pin marked rows exact.
    assert 'MC_VIDEO_STRENGTH: strength' in NODES
    assert 'MC_VIDEO_STRENGTH: strengths[slot - 1]' in NODES
    assert '"strengths":[1,1,1]' in NODES
    assert '_patched_cond_video_rows' in PAYLOAD
    assert '_patched_forward' in PAYLOAD
    assert 'apply_forward_patch' in PAYLOAD
    assert 'apply_cond_video_patch' in PAYLOAD
    assert 'visual_cond_noise_aug"] = 0.999' in PAYLOAD
    assert 'audio_cond_noise_aug"] = 0.999' in PAYLOAD
    assert '_h3mc_active_keyframes' in PAYLOAD
    assert '_h3mc_active_refs' in PAYLOAD
    assert 'MC_VIDEO_STRENGTH' in LAYOUT
    assert 'MC_AUDIO_STRENGTH' in LAYOUT

    # Custom Video takes multiple clips: dynamic video_ and video_audio_
    # slot prefixes declared through the multi-prefix dynamic input map.
    assert '("video_", ("IMAGE",))' in NODES
    assert '("video_audio_", ("AUDIO",))' in NODES
    assert 'video_state' in NODES

    # Timeline UI/backend wiring fixes: linked video audio ignores stale
    # audio_len, and dynamic inputs follow stable clip ids.
    assert ': clip.audio_link' in TIMELINE_JS
    assert 'slot = int(clip.get("id") or idx)' in NODES

    # Duplicate-install hardening lives in the layout patcher.
    assert '_FOREIGN_ORIG_NAMES' in LAYOUT
    assert '_find_dup_installs' in LAYOUT

    print("PASS: multi-ref timeline-audio fork structure + Python syntax")


if __name__ == "__main__":
    main()
