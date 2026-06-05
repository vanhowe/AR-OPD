# Data Layout

This directory contains the lightweight public data assets for the AR-OPD code release.

Included directly:

- `data/manifests/`: asset manifests and checksums.
- `data/packages/math/eval/`: small math evaluation datasets.
- `data/packages/code/eval/`: code evaluation datasets used by `eval_code.py`.
- `data/packages/medical/eval/`: medical QA evaluation JSONL files.
- `data/asft_math_data/eval_data/`: ASFT-aligned math benchmark JSONL files.

Not bundled directly:

- large raw training chunk files under `data/github_chunks/`;
- materialized `data/derived/**` training datasets;
- checkpoints, report bundles, and generated artifacts.

The manifest `data/manifests/github_chunks.json` is retained for provenance and for users who have access to the raw-train chunk bundle. If `data/github_chunks/` is present, `scripts/prepare_data_layout.sh` restores raw train files before validation. If it is absent, the script skips raw-train restoration and validates the lightweight assets included in this release.

Expected full local layout on a training machine:

- `data/packages/math/train/...`
- `data/packages/code/train/...`
- `data/packages/medical/train/...`
- `data/derived/math_dual_100k/...`
- `data/derived/code_100k/...`
- `data/derived/medical_clean_50k/...`

Useful commands from the `code/` directory:

```bash
bash scripts/prepare_data_layout.sh
python scripts/validate_assets.py --manifest data/manifests/assets.json --root .
```

After raw training files are available locally, derived datasets can be rebuilt with:

```bash
bash scripts/materialize_math_10k.sh
bash scripts/materialize_code_10k.sh
bash scripts/materialize_all_derived_datasets.sh
```
