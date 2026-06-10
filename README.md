<h1 align="center">AR-OPD</h1>

<p align="center">
  <strong>Beyond Absolute Imitation: Anchored Residual Guidance for Privileged On-Policy Distillation</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2606.10385">arXiv</a> |
  <a href="https://vanhowe.github.io/AR-OPD/">Project Page</a> |
  <a href="docs/assets/aropd-paper.pdf">Paper</a> |
  <a href="code/">Code</a> |
  <a href="code/data/README.md">Data</a> |
  <a href="RELEASE_STATUS.md">Release Status</a> |
  <a href="ARXIV_CHECKLIST.md">arXiv Checklist</a>
</p>

<p align="center">
  <img src="docs/assets/architecture.png" alt="AR-OPD architecture" width="92%">
</p>

## Overview

AR-OPD studies a failure mode in privileged on-policy distillation: a full-view oracle teacher can provide useful answer-directed signal, but direct imitation may push the student toward tokens that are correct only under unavailable future information. AR-OPD treats this as a target-design problem by decomposing privileged supervision into:

- a **partial-view anchor** that remains locally reachable from the student prefix;
- a **controlled full-minus-partial residual** that transfers destination-directed foresight without turning the full teacher into an absolute target.

The result is a dual-view distillation target that preserves local sequence compatibility while still using privileged information.

## Highlights

| Result | Summary |
| --- | --- |
| Main average | **70.3**, the strongest average across seven benchmarks |
| vs. Full OPD | **+2.3 points** over full privileged OPD |
| vs. Base | **+12.1 points** over the base model |
| Shortcut diagnostic | **21.7% fewer shortcut events** than Full OPD at the final checkpoint |
| Long rollouts | **+7.2 points** over Full OPD on 768-1024 token rollouts |

## Method

Standard privileged OPD treats the full-view teacher distribution as the imitation target. This can be brittle because the teacher is conditioned on future information that the student cannot access at the current prefix. AR-OPD instead constructs:

```text
anchored target = partial-view anchor + lambda * (full-view signal - partial-view signal)
```

This keeps the target tied to a causally reachable partial view while using the full view only as a bounded directional update.

## Results and Diagnostics

<p align="center">
  <img src="docs/assets/training_dynamics.png" alt="AR-OPD validation accuracy, shortcut count, and long-rollout accuracy" width="92%">
</p>

AR-OPD improves validation accuracy, reduces shortcut generation, and is strongest on longer rollouts. The long-horizon result is important because support mismatch is expected to compound as student prefixes drift farther from the privileged oracle trace.

<p align="center">
  <img src="docs/assets/teacher_reliability.png" alt="Teacher reliability and support-gap diagnostics" width="82%">
</p>

The target-reliability diagnostics show that full-view supervision can create larger teacher-student disagreement and support gaps near rollout tails. This supports the central design choice: privileged information should be decomposed and controlled rather than copied as a monolithic target.

## Code

Training and evaluation code is included in this repository under [`code/`](code/).

Useful entry points:

- [`code/main.py`](code/main.py): main training entry point.
- [`code/distil_trainer.py`](code/distil_trainer.py): distillation trainer and AR-OPD target construction.
- [`code/distil_config.py`](code/distil_config.py): method, domain, and training configuration objects.
- [`code/eval_math.py`](code/eval_math.py), [`code/eval_code.py`](code/eval_code.py), [`code/eval_medical.py`](code/eval_medical.py): evaluation entry points.
- [`code/scripts/`](code/scripts): launch, materialization, validation, and result-collection scripts.
- [`code/configs/`](code/configs): hardware, domain, and method configuration files.

Minimal setup path:

```bash
git clone https://github.com/vanhowe/AR-OPD.git
cd AR-OPD/code
bash scripts/install_env.sh
bash scripts/prepare_data_layout.sh

# After raw training files or chunk bundles are available locally:
bash scripts/materialize_all_derived_datasets.sh
```

## Data

This repository includes lightweight evaluation assets, data manifests, and reconstruction scripts rather than large materialized training directories.

Included directly in [`code/data/`](code/data/):

- math, code, and medical evaluation JSONL files;
- ASFT-aligned math benchmark files;
- manifests and scripts for restoring and validating local data.

Kept local by design:

- large raw training chunk bundles;
- materialized `data/derived/**` training datasets;
- checkpoints;
- large artifacts and report bundles.

After cloning the code package, run:

```bash
bash scripts/prepare_data_layout.sh
```

This validates the expected asset layout. Derived datasets can then be rebuilt with the materialization scripts in [`code/scripts/`](code/scripts/).

## Release Status

| Component | Status |
| --- | --- |
| Paper | Included under [`docs/assets/aropd-paper.pdf`](docs/assets/aropd-paper.pdf) |
| Project page | Live at <https://vanhowe.github.io/AR-OPD/> |
| Training and evaluation code | Included under [`code/`](code/) |
| Lightweight evaluation data and manifests | Included under [`code/data/`](code/data/) |
| Large derived datasets and checkpoints | Not bundled; regenerated locally from scripts |
| arXiv link | <https://arxiv.org/abs/2606.10385> |

## Repository Contents

- [`docs/index.html`](docs/index.html): project homepage.
- [`docs/assets/aropd-paper.pdf`](docs/assets/aropd-paper.pdf): current paper PDF.
- [`docs/assets/architecture.png`](docs/assets/architecture.png): AR-OPD method architecture.
- [`docs/assets/training_dynamics.png`](docs/assets/training_dynamics.png): validation accuracy, shortcut count, and long-rollout results.
- [`docs/assets/teacher_reliability.png`](docs/assets/teacher_reliability.png): teacher reliability and support-gap diagnostics.
- [`code/`](code/): training, evaluation, configuration, and data-preparation code.
- [`LICENSE`](LICENSE): Apache-2.0 license for released code and repository assets.
- [`RELEASE_STATUS.md`](RELEASE_STATUS.md): current release status and data/code boundaries.
- [`ARXIV_CHECKLIST.md`](ARXIV_CHECKLIST.md): arXiv release checklist.

## Citation

```bibtex
@misc{aropd2026,
  title = {Beyond Absolute Imitation: Anchored Residual Guidance for Privileged On-Policy Distillation},
  author = {Wenhao Zhang},
  year = {2026},
  eprint = {2606.10385},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2606.10385}
}
```

## Security

Do not commit GitHub tokens, API keys, private checkpoints, private datasets, or local machine paths. If a personal access token has been pasted into any chat, document, issue, or terminal output, revoke it and generate a new one.
