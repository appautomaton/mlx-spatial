# Model Publishing

This page covers converted model bundles, not the Python package.

## Naming

Publish AppAutomaton-first:

```text
appautomaton/<model-name>
```

Use an AppAutomaton model repo first. For SAM3D, the public runtime bundle is:

```text
appautomaton/sam-3d-objects-mlx
```

It includes the converted SAM3D checkpoints plus the converted MoGe pointmap dependency under `moge/`. Defer any `mlx-community` duplicate until the AppAutomaton repo has a stable model card, audit artifact, and working consumer command.

The tracked SAM3D model-card source lives under:

```text
model-cards/sam-3d-objects-mlx/
```

The published SAM3D repo must include the upstream SAM License copy as `LICENSE`, because the license requires a copy of the agreement when distributing SAM Materials or derivatives.

For LiTo, use the research-scoped AppAutomaton repo name:

```text
appautomaton/lito-research-mlx
```

The local upload root is:

```text
weights/lito-research-mlx
```

The tracked model-card source lives under:

```text
model-cards/lito-research-mlx/
```

LiTo is research-only and non-commercial under Apple's model license. The model repository must include `LICENSE_MODEL`, identify the safetensors files as an unofficial converted derivative, and avoid language that implies Apple endorsement.

## What The Model Card Should Contain

Include:

- Source model name and link.
- Source code repo link when relevant.
- Source license name and link.
- Access or gating requirements.
- Conversion command or reproducible conversion summary.
- Expected local file layout.
- Included dependency models and their source links.
- Weight audit summary.
- Compatible `mlx-spatial` version or commit.
- Minimal inference command.
- Known limitations and quality gates.

Do not copy large upstream model-card sections. Attribute and link to upstream instead.

Lead with the user path: install `mlx-spatial`, download the model bundle, validate assets, and run one minimal command. Put upstream attribution and license details after the usage path so humans can start quickly and agents can identify the runnable contract.

For SAM3D, use:

- Source model: https://huggingface.co/facebook/sam-3d-objects
- Source code: https://github.com/facebookresearch/sam-3d-objects
- License: upstream SAM License as linked by the source model/code pages

For LiTo, use:

- Source project: https://apple.github.io/ml-lito/
- Source code: https://github.com/apple/ml-lito
- License: Apple Machine Learning Research Model License Agreement, `LICENSE_MODEL`
- Local model-card source: `model-cards/lito-research-mlx/README.md`

## Audit Artifact

Keep a machine-readable audit with the converted model bundle, for example:

```text
weights/sam-3d-objects-mlx/weight-audit-source-vs-mlx.json
```

The audit should include:

- source root and converted root
- compared divisions or checkpoint roles
- tensor counts
- missing tensors
- extra tensors
- shape mismatches
- dtype conversion notes
- maximum absolute difference

The SAM3D audit should compare `ss_generator`, `slat_generator`, `ss_decoder`, `slat_decoder_gs`, `slat_decoder_mesh`, and `slat_decoder_gs_4`, and report tensor counts plus missing, extra, shape-mismatch, dtype, and numeric-difference summaries.

## License And Access Checks

Before publishing or duplicating a converted model bundle:

1. Re-open the source model page and source repository.
2. Confirm the current license terms.
3. Confirm whether the source model is gated.
4. Confirm redistribution rights for converted weights.
5. Record the check date in the model card.

If redistribution is unclear, do not publish converted weights. Publish conversion instructions instead.

Repository visibility must match the card links. If the card links to `mlx-spatial` source, confirm the GitHub repo is public before publishing the model card.

## mlx-community Deferral

Only duplicate to `mlx-community` after:

- the AppAutomaton model repo is published and usable;
- `mlx-spatial` can load it with a documented command;
- the model card includes source, license, conversion, and audit details;
- redistribution terms have been checked;
- the duplicate can trace back to the AppAutomaton repo.
