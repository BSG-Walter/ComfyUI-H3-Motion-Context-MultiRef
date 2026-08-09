# Put this on GitHub

For a true GitHub fork with upstream history:

1. Open https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context
2. Click **Fork** and create the fork under your account.
3. Clone your fork locally.
4. Copy `nodes.py`, `patch_layout.py`, `README.md`, `MODIFICATIONS.md`, and `patches/` from this bundle into it. (`patch_payload.py` and `__init__.py` are included here for a complete install snapshot but are unchanged from upstream.)
5. Keep the upstream `LICENSE` / GPL-3.0 notice.
6. Commit and push.

Or apply only the code patch:

```bash
git apply patches/multi_ref_timeline_audio.patch
git add nodes.py patch_layout.py
git commit -m "Add Ref2VA multi-ref timeline-audio compatibility"
git push
```

The README/MODIFICATIONS files should also be copied so the modified fork is clearly identified and attributed.
