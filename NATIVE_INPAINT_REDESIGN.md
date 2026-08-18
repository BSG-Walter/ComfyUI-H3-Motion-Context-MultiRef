# H3 Timeline: Native Denoise-Mask Redesign (handoff notes)

**Goal:** Replace the post-CFG hard-clamp hack with ComfyUI's native
MiniMax-H3 per-token video/audio noise-mask machinery
(`Comfy-Org/ComfyUI` commit `ff6c8a8a`, PR #15375), which is present in the
installed ComfyUI (verified `git merge-base --is-ancestor ff6c8a8 HEAD`).
Context commit (native `MiniMaxH3AddGuide`): `e01fb4c5`.

---

## 1. What commit ff6c8a8 added (installed at .../ComfyUI/comfy/)

### comfy/ldm/minimax/model.py
- `mask_row_values(mask, latent_t, lat_h, lat_w)`:
  - input `[T, H, W]` (already pooled to token grid, dims = *padded* latent
    dims), replicate-pads to `(lat_h, lat_w)`, reshapes to
    `(latent_t, lat_h//2, 2, lat_w//2, 2)`, `amax` over the patch dims →
    one value per DiT token (`T * lat_h/2 * lat_w/2`), `None` if all values
    >= `1 - 1e-3` (nothing masked).
- `MiniMaxH3Model.forward/_forward` accept `denoise_mask=None,
  audio_denoise_mask=None`:
  - video: `m = mask_row_values(denoise_mask[0,0], latent_t, lat_h, lat_w)`
    → per-token mask `m`; row timestep `rows_t = (1 - m*sigma_v).clamp(max=t_pin_v)`
    (`t_pin_v = max(t_v, VISUAL_COND_TIMESTEP=0.999)`).
  - audio: `m = audio_denoise_mask[0,0].flatten()` (length `2*T_audio`);
    `sigma_a = 1 - t_a`; `rows_t = (1 - m*sigma_a).clamp(max=t_pin_a)`
    (`t_pin_a = max(t_a, AUDIO_COND_TIMESTEP=1.0)`).
  - if all rows share one value → whole-stream `seg_t` gets it; otherwise
    per-row timestep table: `unique_t` extended with the per-row values,
    `rows_to_mod_index(rows_t, tag)` maps values → mod-row indices
    (`t_row[v]*3 + tag`), segments get per-row tensors. Both the DiT blocks
    (adaln/gate mod-rows) and the `final_layer` (via `_mod_row`) consume
    them.
  - Net effect: **a masked row runs at sigma = m·sigma_stream. m=0 ⇔ row at
    cond strength (0.999 / 1.0) = "hold clean content". m=1 ⇔ normal
    generation. 0<m<1 ⇔ intermediate schedule.**
- `_sliding_window` row indexing changed to accept per-token index tensors.

### comfy/model_base.py (MiniMax BaseModel)
- `extra_conds(**kwargs)` now reads `denoise_mask` and merges
  `_denoise_mask_conds(...)` output (only when `latent_shapes` has ≥ 2
  streams i.e. packed AV).
- `_pool_masks_to_token_grid(masks)`:
  - `masks[0]` (video, `[B,24,T,H,W]`): pads h/w to patch multiples
    (`patch_size (1,2,2)` → both even), `amax` per 2×2 patch → token grid
    `[B,24,T,H/2,W/2]`, then `repeat_interleave` back up and crop
    (the returned *cond* is only `[:, :1]`, `[1,1,T,H/2,W/2]`).
  - `masks[1]` (audio, `[B,32,2,T]`): `amax(dim=1, keepdim=True)` over the
    32 channels → `[B,1,2,T]`.
- `_token_grid_masks`: `unpack_latents` then
  `torch.ceil(mask*256.0)/256.0` quantize.
- `_denoise_mask_values`: emits `denoise_mask` cond only if
  `amin(masks[0]) < 1-1e-3` (`masks[0][:1,:1]` clone), emits
  `audio_denoise_mask` only if `amin(masks[1]) < 1-1e-3`
  (`masks[1][:1].amax(dim=1, keepdim=True)`, i.e. `[1,1,2,T]`).
- `scale_latent_inpaint(sigma, noise, latent_image, x=None, denoise_mask=None, **kwargs)`:
  the injection. `shapes = self.latent_shapes` (set by the sampler).
  - `cleans = unpack_latents(latent_image, shapes)`,
    `noises = unpack_latents(noise, shapes)`;
  - **video:** `cleans[0] = VISUAL_COND_TIMESTEP*cleans[0] + (1-aug)*noises[0]`
    (injected content is noise-mixed at cond strength);
  - **audio:** if `audio_scale() != 1.0`, rescale by
    `factor = (sigma_v/sigma_a)/audio_scale` so the model sees it clean;
  - repack = `injected`;
  - if `x`/`denoise_mask` given: blended output
    `injected + x_blend_weight·(x - injected)` with
    `x_blend_weight = (token_grid_mask - denoise_mask)/(1-denoise_mask).clamp(1e-6)`
    (0 where mask=1, 1 where mask=0), `token_grid_mask = pack(_token_grid_masks)`.

### comfy/samplers.py (CFGGuider.__call__, line ~636)
- `x = x*denoise_mask +
  inner_model.scale_latent_inpaint(x=x, sigma=sigma, noise=self.noise,
  latent_image=self.latent_image, denoise_mask=denoise_mask) * latent_mask`
- after the model: `out = out*denoise_mask + self.latent_image*latent_mask`
  ⇒ **mask=0 regions come out exactly = `latent_image`** (the clean latent we
  supply); mask=1 regions = pure prediction.

---

## 2. Verified shapes / constants (current core)

| Thing | Value |
|---|---|
| Video latent | `[B, 24, T_v, H/16, W/16]` (24 channels! not 16) |
| Audio latent | `[B, 32, 2, T_a]` (2 frames per latent token, T_a ≈ frames·5/3) |
| AV latent type | `comfy.nested_tensor.NestedTensor((video, audio))`, `is_nested=True` |
| FRAME_PER_TOKEN | `(1,4,4,4,4)`; 17 pixel frames = 5 tokens; pixel boundaries at token t: `cum[t]` (0,1,5,9,13,17,18,22,...) |
| Latent length | `frame_count = 5+17k`; `latent_t = 5k+2`; chunk c (pixel `17c`) = tokens `[5c, 5c+5)` |
| FRAME_RESCALE | `5.0/3.0` audio latent frames per pixel frame (40 Hz @ 24 fps) |
| VISUAL_COND_TIMESTEP | 0.999 (video cond pin) |
| AUDIO_COND_TIMESTEP | 0.999→actually 1.0 in source (line 33) |
| Video mask (user gives) | `[B, 1, T_v, H_lat, W_lat]`, 0=hold content, 1=generate |
| Audio mask (user gives) | `[B, 1, 2, T_a]` (channel-major like `pack_audio`: ch0 t0..T-1, ch1 t0..T-1) |
| Segment "audio" | `2·T_a` rows (2 per latent token) → flatten of `[0,0]` of `[1,1,2,T]` aligns exactly |
| Sampler mask reshape | `comfy.utils.reshape_mask`: interpolate to latent res, repeat channels to 24/32, batch repeat |
| `self.latent_shapes` on model | set at `samplers.py:1221` (`inner_sample`) from the nested latent |

Mask semantics (native): **value m ∈ [0,1]: 1 = generate, 0 = preserve
exact (cond-pinned timestep + injected clean latent), fractional = blend.**
Old clamp semantics: strength st → `(1-st)·pred + st·clean`. These are
equivalent when **mask_value = 1 - strength** (out = m·pred + (1-m)·clean,
plus native per-row timestep modulation as a bonus).

---

## 3. Full data flow (native, no patches)

```
node ──▶ LATENT {"samples": NestedTensor([clean_video, clean_audio]),
                 "noise_mask": NestedTensor([video_mask, audio_mask])}
              └─▶ KSampler (latent_image input; denoise=1.0)
  comfy/sample.sample() → samplers.sample (line 1276):
    - nested latent → latent_shapes = [v_shape, a_shape]; pack→[B,1,24THW+64Ta]
    - nested mask → unbind → per-stream reshape_mask → pack masks
  → CFGGuider.__call__ (line 634): scale_latent_inpaint each step (inject)
    → model(x, sigma, denoise_mask=packed) F
  → MiniMax extra_conds: _denoise_mask_conds → denoise_mask/audio_denoise_mask
    conds (token-grid, only if amin<1-1e-3)
  → MiniMaxH3Model._forward: mask_row_values → per-token rows_t → t_row table
    → mod_segments/rows → DiT blocks + final_layer per-row timesteps
  → out = pred*mask + latent_image*(1-mask)   (CFGGuider, exact preserve)
```

No model patching, no `sampler_post_cfg_function` needed. The conditioning
path (`minimax_keyframes` keyframes / refs → cond rows pinned at cond
timestep) is orthogonal and stays.

---

## 4. Implementation plan (nodes.py)

Keep: conditioning output (keyframes), media loading/decode, grid-align
encode machinery (`_grid_segment`, `_pad_clip`, `_video_token_window`,
`_video_token_window`, `_frame_to_audio_latent_idx`, audio two-pass
encode). Replace: clamp hook + post-CFG.

1. **New output** `inpaint latent` (LATENT), appended after `model` — old
   edges keep working (RETURN_TYPES grows to 3). Model output becomes pure
   passthrough (no hook) for workflow compat. clampspecs/clamp_strength
   removed; `clamp_strength` input dropped (or kept as no-op alias? drop —
   cleaner; tooltip mention masks).
2. Build per-clip, exactly like today:
   - video clip: `seg_start, lead, seg_len = _grid_segment(rel, want, fc)`;
     encode padded seg; `i0, i1 = _video_token_window(lead, want)`;
     target token span = `s = (seg_start//17)*5 + i0`, `e = s + (i1-i0)`
     (verified: token index of pixel boundary `17c` is `5c`, both on the
     encode side and the target grid — the segment padded to a 17-boundary
     lattice maps 1:1).
   - inject: `clean_video[:, :, s:e] = enc_latent[:, :, i0:i1]`
     (window `i0..i1` INCLUDING boundary token — better than the old
     `i0c=i0+1` skip; boundary tokens are lead/tail repeats = hold frames).
   - mask: `video_mask[:, :, s:e] = min(video_mask[:,:,:,s:e], mask_val)`,
     `mask_val = 1 - strength` (min-reduce across overlapping clips).
   - audio: token start `sa = _frame_to_audio_latent_idx(frame, fps)`;
     `clean_audio[..., sa:sa+rt] = audio_lat`;
     `audio_mask[..., sa:sa+rt] = min(..., 1 - audio_strength)`.
   - image clip: same as video clip with want_len; full token window.
3. Outputs:
   ```python
   clean_video = torch.zeros([1, 24, latent_t, H, W])        # 24 channels!
   clean_audio = torch.zeros([1, 32, 2, audio_latent_t])
   video_mask = torch.ones([1, 1, latent_t, H, W])
   audio_mask = torch.ones([1, 1, 2, audio_latent_t])
   out_latent = {"samples": NestedTensor([clean_video, clean_audio]),
                 "noise_mask": NestedTensor([video_mask, audio_mask])}
   return (out_cond, model, out_latent)
   ```
4. KSampler usage: plug `inpaint latent` into the sampler's latent input,
   denoise=1.0 (mask handles the rest); steps/scheduler unchanged.

Gotchas (verified against core):
- Video channels are **24**, not 16 — the old `_create_h3_clamp_hook`
  didn't care, but the composite does.
- Audio mask dim order `[1,1,2,T]` matches `pack_audio`'s channel-major
  row order (ch0 all tokens, then ch1).
- `latent_shapes` must be ≥2 streams for the masks to apply at all; the
  nested latent guarantees that.
- Masks use amax pooling: a token is preserved if ANY of its 2×2 pixels is
  mask<1; build masks full-token (full-frame clips: whole row slice), no
  partial-pixel masks needed.
- If the workflow wants 100% exact clip content: strength=1.0 (mask 0).
- If `model` input absent, node still works for conditioning + latent.
- `denoise` must be 1.0 (full) — img2img-style partial denoise mixes
  `latent_image` everywhere, defeating the mask.

## 5. Verification checklist (after editing nodes.py)

1. `python -c "import importlib, nodes; from nodes import MiniMaxH3Timeline"` —
   imports clean (custom node loads on ComfyUI restart).
2. Tests: run `tests/test_timeline_node.py` (uses fake VAEs; update expected
   outputs where the hook/strength behavior changed).
3. Manual workflow: EmptyMiniMaxH3LatentAV → timeline (image + video clip +
   audio clip) → KSampler(latent = new inpaint output, denoise 1.0, ~30
   steps) → decode. Assert: clip frames identical to source (mask 0),
   audio track starts at frame anchor, non-masked region coherent.
4. Edge cases: clip at frame 0, clip exactly at frame_count-want, two
   overlapping clips, audio-only clip, strength=0.5, GIF/media files.

## 6. Notes re: native AddGuide (e01fb4c5)

`comfy_extras/nodes_minimax_h3.py` `MiniMaxH3AddGuide` only appends
`minimax_keyframes` (conditioning). It does NOT inject — no denoise masks.
So nothing to sync with; our node remains the only one doing exact
injection. `EmptyMiniMaxH3LatentAV` shows canonical nested-latent shape
(`_empty_av_latent`, video 24 ch, audio 32×2×T).