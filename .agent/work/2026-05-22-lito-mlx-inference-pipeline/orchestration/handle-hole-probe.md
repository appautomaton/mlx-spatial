# LiTo Handle-Hole Probe

Date: 2026-05-23

## Scope

This is a static/local probe for the remaining teacup handle void issue. It does not run CUDA, import vendor runtime code, or change the accepted LiTo implementation gate.

## Findings

1. The accepted uncapped PLY is structurally sane, so the handle issue is not a generic export corruption symptom. `scripts/lito/inspect_quality.py` reported `1108288` vertices, `62` properties, finite fields, no flags, opacity probability median `0.056885`, scale exp median `0.004650`, and quaternion norm median `1.000000` for `outputs/lito/teacup-quality-crop-uncapped.ply`.

2. The teacup input alpha is not a tight object matte. Local alpha inspection of `inputs/trellis2/teacup.png` found image size `(2648, 2361)`, `1977753` pixels at alpha `0`, `4172106` pixels at alpha `255`, and the largest `alpha > 204` connected component has bbox `(1, 1, 2648, 2360)` and touches the image edges. That means the current crop path sees almost the full canvas as foreground, so the visible handle-hole/background boundary is not reliably isolated before conditioning.

3. Upstream LiTo uses the same broad alpha shortcut before `rembg`: if an RGBA input has any non-255 alpha, it uses that alpha instead of running background removal (`vendors/ml-lito/demos/lito/fastapi_lito_demo.py:157-178`). For this specific input, a better alpha-quality heuristic or connected-component cleanup may be needed even though upstream source has the same shortcut.

4. Quaternion convention remains a secondary audit item, not the first handle-hole target. Apple export passes `gs_dict["quaternion"]` directly to `Gaussians(...).save_ply(...)`, and `plibs.gs_utils.build_rotation` treats `rot_0` as scalar `w` despite a stale `4xyzw` comment. The local checkpoint-backed PLY export preserves raw `rot_0..rot_3`; the only suspicious local behavior is the zero-norm fallback identity using slot `3`. Real generated quaternions are not near zero in the inspector, so this is unlikely to explain the handle void, but it is worth a small convention test/fix later.

## Likely Next Slice

Do not mix this into the binary PLY storage commit. The next quality slice should be a preprocessing/matte slice:

- add an alpha-quality diagnostic for RGBA inputs whose high-alpha component touches image edges or covers too much canvas;
- optionally run the existing MLX RMBG path, or add a conservative connected-component/matte cleanup when RGBA alpha is present but not object-like;
- regenerate teacup with `--max-init-coords-per-batch none` and compare the handle void in the same KIRI/3DGS viewer.

If the matte-cleanup run does not improve the handle, the next probe should be quaternion convention and local renderer/export parity.

## Commands

```bash
UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run python -c '<alpha connected-component probe>'
python3 -m json.tool /tmp/lito-teacup-quality-crop-uncapped.json
```
