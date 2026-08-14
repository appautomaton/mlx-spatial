# BRIA RMBG-2.0 License and Attribution

This distribution includes a selectively quantized derivative of BRIA
RMBG-2.0.

- Original work: **BRIA RMBG-2.0**
- Creator: **BRIA AI**
- Model: <https://huggingface.co/briaai/RMBG-2.0>
- Source: <https://github.com/Bria-AI/RMBG-2.0>
- License: [Creative Commons Attribution-NonCommercial 4.0
  International](https://creativecommons.org/licenses/by-nc/4.0/)

The original model is available for non-commercial use under CC BY-NC 4.0.
Commercial use requires a separate agreement with BRIA. Refer to the original
model page for BRIA's current commercial licensing information.

## Changes in This Distribution

Eligible attention and MLP matrix weights were repacked for direct MLX affine
8-bit execution with group size 64. Other weights remain in their source
precision. The derivative preserves the original logical tensor names through
checkpoint metadata and is distributed as part of the TRELLIS.2 MLX 8-bit
runtime bundle.

No endorsement by BRIA AI is implied. This notice summarizes the source,
attribution, and changes; the linked CC BY-NC 4.0 terms remain controlling.
