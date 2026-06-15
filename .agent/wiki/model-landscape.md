# 3D Generative Model Landscape (reference)

External reference informing which models mlx-spatial targets and ports. **May-2026 snapshot — re-search before relying; the field moves fast.**

## Field shape

Open-weight 3D-gen is dominated by **sparse-voxel + flow-matching-DiT + separate PBR-texture-diffusion** stacks (representation trajectory NeRF→triplane→GS→VecSet→sparse-voxel). Quality leaders: TRELLIS.2 (Microsoft/Tsinghua, O-Voxel, MIT), the Hunyuan3D 2.x family (Tencent, ShapeVAE+PBR), TripoSG/TripoSF (VAST), SAM 3D (Meta), Ultra3D. World-scale: HY-World 2.0 (Tencent); NVIDIA Lyra 2.0 (research-only).

mlx-spatial targets TRELLIS.2 / HY-World 2.0 / SAM 3D. The gap it fills: **no production-quality open-weight 3D generator runs on Apple Silicon / MLX** as of 2026. Field-wide unsolved gaps relevant here: thin structures (hair/cables/strings — cf. our thin-feature defect), production-ready topology (clean quads), PBR/lighting disentanglement.

## LiTo (Apple) — strongest candidate next MLX port

LiTo (Apple ML Research, ICLR 2026, arXiv 2603.11047): single-image → 3DGS with view-dependent appearance via Surface Light Field Tokenization (RGBD surface samples → 8192×32 tokens → flow-matching DiT). Strong fit because **Apple already ships a working MLX inference backend upstream** (github.com/apple/ml-lito, ~160 s/img on M4 Max) — not a from-scratch port. Public weights at ml-site.cdn-apple.com/models/lito/ (`lito_new.ckpt` tokenizer + `lito_dit_rgba.ckpt` generator recommended); read `LICENSE_MODEL` before redistributing converted safetensors. Not currently being pursued.
