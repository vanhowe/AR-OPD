# AR-OPD Code

This directory contains the public training, evaluation, configuration, and data-preparation code for:

**Beyond Absolute Imitation: Anchored Residual Guidance for Privileged On-Policy Distillation**

The release includes runnable source code, launch scripts, configuration files, lightweight evaluation data, data manifests, and asset-validation utilities. Large derived datasets, checkpoints, generated reports, and raw training chunk bundles are not bundled.

Implementation note: legacy `gdsdft_*` flags and method configs correspond to the AR-OPD anchored residual target used in the paper. Paper runs use the prompt and completion limits specified by the launch scripts and experiment tables; `configs/base.env` keeps a larger general default completion length for local experimentation.

## Structure

| Path | Purpose |
| --- | --- |
| `main.py` | Main training entry point |
| `distil_trainer.py` | Distillation trainer and target construction logic |
| `distil_config.py` | Domain, method, and training configuration objects |
| `cad_utils.py` | Shared helper utilities |
| `eval_math.py` | Math evaluation entry point |
| `eval_code.py` | Code evaluation entry point |
| `eval_medical.py` | Medical QA evaluation entry point |
| `scripts/` | Launch, materialization, validation, and result-collection scripts |
| `configs/` | Hardware, method, and domain configuration files |
| `data/` | Lightweight evaluation assets, manifests, and data-layout documentation |

## Setup

From the repository root:

```bash
cd code
bash scripts/install_env.sh
bash scripts/prepare_data_layout.sh
```

`prepare_data_layout.sh` validates the lightweight assets included in this release. If a local `data/github_chunks/` directory is present, it also restores raw-train files before validation.

## Data

Included directly:

- math evaluation assets;
- code evaluation JSONL files;
- medical QA evaluation JSONL files;
- ASFT-aligned math benchmark JSONL files;
- data manifests and checksums.

Not bundled:

- raw training chunk bundles;
- materialized `data/derived/**` training datasets;
- checkpoints;
- generated report bundles.

See [`data/README.md`](data/README.md) for the full layout.

## Materialization

After raw training files or chunk bundles are available locally, derived datasets can be rebuilt with:

```bash
bash scripts/materialize_math_10k.sh
bash scripts/materialize_code_10k.sh
bash scripts/materialize_all_derived_datasets.sh
```

The materialization scripts expect local model/tokenizer paths to be configured through environment variables such as `MODEL_NAME` and `TOKENIZER_MODEL_NAME` when needed.

## Training

Main experiment launchers:

```bash
bash scripts/launch_main_experiments.sh
bash scripts/launch_full_suite.sh
```

Focused ablation launchers:

```bash
bash scripts/launch_10k_lambda_sweep.sh
bash scripts/launch_10k_partial_rate_sweep.sh
```

Useful overrides:

- `DOMAINS="math code"` to choose domains.
- `PARTIAL_RATES="0.25 0.50 0.75"` to control partial-rate sweeps.
- `MATERIALIZE_ONLY=1` to prepare datasets without launching training.
- `SKIP_BASELINES=1` or `SKIP_SWEEP=1` to run one side of an ablation.

## Evaluation

Math:

```bash
RUN_DIR=/path/to/checkpoint_run \
RUN_NAME=my_run \
bash scripts/run_math_eval_benchmarks.sh
```

Code:

```bash
python eval_code.py --help
```

Medical QA:

```bash
python eval_medical.py --help
```

## Notes

- The default launch scripts target multi-GPU training and should be adapted to the local hardware configuration under `configs/hardware/`.
- Large checkpoints and artifacts should be written outside the Git repository.
- The code is released under the repository-level Apache-2.0 license.
