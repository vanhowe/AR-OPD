# Release Status

This repository is the public release hub for AR-OPD.

## Current Status

| Component | Status | Location |
| --- | --- | --- |
| Project page | Live | https://vanhowe.github.io/AR-OPD/ |
| Paper | Published on arXiv | https://arxiv.org/abs/2606.10385 |
| Paper PDF copy | Included | `docs/assets/aropd-paper.pdf` |
| Paper source | Submitted through arXiv | source package maintained separately |
| Training code | Included | `code/` |
| Evaluation code | Included | `code/eval_math.py`, `code/eval_code.py`, `code/eval_medical.py` |
| Launch scripts | Included | `code/scripts/` |
| Config files | Included | `code/configs/` |
| Lightweight eval data | Included | `code/data/` |
| Raw training chunk bundles | Not bundled | restore locally if available |
| Derived datasets | Not bundled | regenerate locally |
| Checkpoints and reports | Not bundled | regenerate locally |
| Code license | Included | Apache-2.0 in `LICENSE` |
| arXiv link | Live | https://arxiv.org/abs/2606.10385 |

## Notes

- `code/scripts/prepare_data_layout.sh` validates bundled lightweight assets.
- If `code/data/github_chunks/` is present locally, the same script restores raw-train files before validation.
- Full dataset materialization requires local access to the raw training files or chunk bundle.
