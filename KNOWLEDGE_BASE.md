# Knowledge Base — ComfyUI-H3-Motion-Context-Timeline

Session notes for this fork. Purpose: don't re-run the empirical investigation.

## Environment

- Node root: `custom_nodes\ComfyUI-H3-Motion-Context-Timeline` (fork of
  seitanism/ComfyUI-H3-Motion-Context-MultiRef, itself a fork of
  NikoDemon80/ComfyUI-H3-Motion-Context).
- Python: `C:\Users\Walter\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe`
  (needs torch + importable `comfy.ldm.minimax`; the bare python on PATH
  will NOT work).
- Tests:
  - `python tests\test_multi_ref_patch_structure.py` (static, plain python).
  - `python tests\test_custom_video_runtime.py` (runtime, with the venv).
  - `node --check js\h3_dynamic_ui.js js\h3_custom_video.js js\h3_custom_audio.js`.
- All tests green as of 2026-08-11.

## Fork architecture

The fork does NOT modify ComfyUI on disk: it is a runtime-patch custom node.

- `patch_layout.py` — `PackedLayout._patched_init`: adds keyframe blocks
  (`minimax_keyframes`, with `MC_KEY`, `MC_VIDEO_STRENGTH`) and refs
  (`minimax_refs`, with `MC_AUDIO_KEY`, `MC_AUDIO_STRENGTH`) to the
  transformer's token layout, with `position_ids` identical to what the
  model saw in training. `_patched_extra_conds` concatenates instead of
  overwriting (stock overwrites `cond_video_latents` when both refs and
  keyframes exist). Constants: `MC_KEY`, `MC_AUDIO_KEY`,
  `MC_AUDIO_STRENGTH`, `MC_VIDEO_STRENGTH`, `FRAME_PER_TOKEN`
  (512/384/288/208/160), `FRAME_RESCALE`. Self-test inside `apply_patch()`.
- `patch_payload.py` — runtime patches for the model:
  - `_patched_cond_video_rows` / `_patched_cond_audio_rows`: the row
    packers. Marked rows are pinned CLEAN (exact, unmixed); unmarked rows
    keep the stock path byte-for-byte.
  - `_patched_forward` (wraps `MiniMaxH3Model.forward`): forces the claims
    `visual_cond_noise_aug` / `audio_cond_noise_aug` to 0.999 while blocks
    are pinned, and does the FLIP: when `t >= s` the block is dropped from
    the ACTIVE SET (`_h3mc_active_keyframes` / `_h3mc_active_refs`, the
    same dict objects) and the reduced layout is rebuilt once per set
    change. The payload lists (keyframes/refs/cond_*_latents) are NEVER
    mutated; the patched packers read latents off the dicts and skip
    inactive blocks, so rows always match the layout by construction (the
    old snapshot+list-reduction bookkeeping desynced on multi-flip runs and
    crashed with 1200-vs-0 / 320-vs-304 shape mismatches; the repro
    `tests/repro_flip_crash.py` pins the fix).
  - `_install` with adoption/takeover (`_ORIG`, `_FOREIGN_ORIG_NAMES`):
    idempotent, coexists with the upstream package if installed.
- `nodes.py` — `MiniMaxH3CustomVideo` (multi-slot, per-slot strength),
  `MiniMaxH3CustomAudio`, `_DynamicInputs`,
  `_ensure_h3_runtime_patches()` installs layout + extra_conds +
  cond-audio + cond-video + forward.
- `js\h3_dynamic_ui.js`, `js\h3_custom_video.js`, `js\h3_custom_audio.js` —
  dynamic inputs and strength widgets.

## How the model works internally (the hard-won part)

- `comfy/model_base.py` `MiniMaxH3.extra_conds` (~lines 2109-2180) builds
  the payload: `minimax_keyframes`, `minimax_refs`, `minimax_frame_count`,
  `cond_video_latents`, `cond_audio_latents`, `layout`, `seed`; reads the
  kwargs `minimax_visual_cond_noise_aug` and `minimax_audio_cond_noise_aug`.
- `comfy/ldm/minimax/model.py` `MiniMaxH3Model` (~lines 402-650):
  `forward` -> `_forward` (~line 513) receives the payload in
  `minimax_payload`. `_forward` assigns each token group its timestep /
  noise field: `seg_t["text"] = t_v`, `seg_t["video"] = t_v`,
  `seg_t["audio"] = t_a`, `seg_t["cond"] = max(t_v, vis_aug)`,
  `seg_t["ref_img"] = max(t_v, vis_aug)`, `seg_t["ref_audio"] =
  max(t_a, aud_aug)`.
- **THE CLAIM IS THE MODEL'S TRUTH**: the model learns a condition's noise
  level from the timestep assigned to its tokens. If the rows carry
  `(1-s)` noise but the claim says 0.999, the model treats the noise as
  real content and bakes it into the output. NEVER show noisy rows under a
  clean claim.
- `t_v = 1 - sigma` (sigma = timestep/1000). `t_a = 1 -
  time_shift_sigma(sigma_v, shift_v, shift_a)` with `shift_v=12`,
  `shift_a=3` (the audio timeline runs ~always ahead: at sigma 0.8 ->
  t_a~0.5; at sigma 0.2 -> t_a~0.94).
- `VISUAL_COND_TIMESTEP = 0.999`, `AUDIO_COND_TIMESTEP = 1.0` (stock).
  Stock: `if aug < 1.0:` mixes `aug * z + (1 - aug) * noise` with noise
  from a per-block generator; `aug >= 1.0` -> clean rows. `aug=0.0` ->
  pure noise (NEVER: pure-noise rows = garbage in the output).
- The layout is cached in the payload but rebuilt per step when the
  composition changes (different signature -> stock accepts it; same
  signature -> cache is used).
- ADD SEMANTICS: keyframes ADD extra tokens; the full video latent is
  always present (`img_update` True = every latent position). The FLIP is a
  REMOVAL of tokens; output comes only from the `video`/`audio` layout
  segments (`final_layer`), so dropping tokens is safe.
- `patchify_video` / `unpatchify_video` / `pack_audio` / `unpack_audio` —
  the layout row packers.
- Ref filtering: the UI marks `motion_context_audio` so `_DynamicInputs`
  excludes the ref from "required" and it doesn't get dropped.

## The 3 strength mechanisms tried, and their verdict

1. **Fixed mixing (per-block noise-aug)** — `s * clip + (1-s) * noise`
   under the stock claim. VERDICT: BAD. The lying claim bakes the noise in:
   visible grain in video below ~0.75; audio noise on low-denoise runs.
2. **Per-block noise-aug + honest claim (forcing the aug in `_forward`)** —
   VERDICT: BAD (same underlying issue: the model never saw "slightly
   noisy" rows in training; it either learns the noise pattern or ignores
   it).
3. **PIN-THEN-FLIP (the one that works)** — while `t < s`: CLEAN rows
   pinned exact + claim 0.999 (the canonical image-to-video pair the model
   DID see in training). When `t >= s`: the block's tokens are REMOVED from
   the layout -> the model's own stream covers the region with no
   reference. VERDICT: video validated by the user as PERFECT.

## Lessons learned

- The claim must ALWAYS tell the truth (0.999 or 1.0 for clean rows).
- `aug=0.0` (pure noise) never; `aug=1.0` is the replicated stock boundary
  in the packers.
- Strength clamps in the nodes: `0.0..1.0` (1-based mode).
- Runs without keyframes / weak refs pass through the wrapper untouched
  (payload untouched) — verified by test.
- BEWARE `dict.get(key, default)` when the default is a model attribute:
  Python evaluates the default ALWAYS (AttributeError if the attribute
  doesn't exist). Safe pattern: `x = d.get(k); if x is None: x = self.attr`.
  (Real bug found with `sigma_shift_video` in the test.)

## Status (2026-08-15)

- Hard Clamping architecture refactored for native ComfyUI coexistence via `ModelPatcher` and `sampler_post_cfg_function`.
- Full design, mathematics, and non-invasive integration details documented in `HARD_CLAMPING_ARCHITECTURE.md`.

## Where to continue

- Validate audio with a real run (several strengths, incl. low denoise).
- If the flip cut is noticeable (hard jump mid-sequence), smooth it:
  ideas — an honest mixing ramp near the cut, or overlapping windows.
- Low denoise + exact pin: pinning at claim 0.999 is still correct (video
  proved it), do NOT go back to mixing.
