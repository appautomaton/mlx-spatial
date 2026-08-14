# Licenses for the TRELLIS.2 MLX 8-bit Bundle

This repository combines derivatives of three separately licensed models. No
single license replaces the component licenses below.

| Component | Files | Governing terms |
| --- | --- | --- |
| Microsoft TRELLIS.2-4B | `ckpts/`, `pipeline.json`, `texturing_pipeline.json` | [MIT License](LICENSE_TRELLIS2) |
| Meta DINOv3 ViT-L/16 | `dinov3/` | [DINOv3 License](LICENSE_DINOV3.md) |
| BRIA RMBG-2.0 | `rmbg/` | [CC BY-NC 4.0 attribution and change notice](LICENSE_RMBG2.md) |

The quantization metadata, packaging, and model card describe changes made for
`mlx-spatial`; they do not relax or replace any upstream term. You must comply
with every license that applies to the files you use or redistribute.

Because the bundle includes RMBG-2.0, use of the complete bundle is limited to
non-commercial purposes unless you obtain separate commercial rights from
BRIA. The DINOv3 agreement also includes redistribution, trade-control, and
other conditions that must be read in full.

This repository is an unofficial derivative and is not affiliated with or
endorsed by Microsoft, Meta, BRIA, or the original authors.

## Upstream References

- TRELLIS.2-4B: <https://huggingface.co/microsoft/TRELLIS.2-4B>
- TRELLIS.2 source: <https://github.com/microsoft/TRELLIS.2>
- DINOv3 ViT-L/16: <https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m>
- DINOv3 source: <https://github.com/facebookresearch/dinov3>
- RMBG-2.0: <https://huggingface.co/briaai/RMBG-2.0>
- RMBG-2.0 source: <https://github.com/Bria-AI/RMBG-2.0>
