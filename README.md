# AR-OPD

Project page and public resources for:

**Beyond Absolute Imitation: Anchored Residual Guidance for Privileged On-Policy Distillation**

AR-OPD studies a failure mode in privileged on-policy distillation: a full-view oracle teacher can provide useful answer-directed signal, but direct imitation may push the student toward tokens that are correct only under unavailable future information. AR-OPD addresses this as a target-design problem by decomposing privileged supervision into a locally reachable partial-view anchor and a controlled full-minus-partial residual.

## Project Page

The project page is live at:

**https://vanhowe.github.io/AR-OPD/**

The page includes the paper draft, method figure, target-reliability diagnostics, and main result highlights.

## Highlights

- **+2.3 points** over full privileged OPD on the main benchmark average.
- **+12.1 points** over the base model.
- **21.7% fewer shortcut events** in the NuminaMath training diagnostic.
- **+7.2 points** over Full OPD on 768-1024 token rollouts.

## Resources

- Paper draft: [`docs/assets/aropd-paper.pdf`](docs/assets/aropd-paper.pdf)
- Project page source: [`docs/index.html`](docs/index.html)
- Method figure: [`docs/assets/architecture.png`](docs/assets/architecture.png)
- Target-reliability diagnostics: [`docs/assets/teacher_reliability.png`](docs/assets/teacher_reliability.png)
- Training dynamics and long-rollout results: [`docs/assets/training_dynamics.png`](docs/assets/training_dynamics.png)
- Code package: [GD-Train-Collab](https://github.com/vanhowe/GD-Train-Collab)

## Citation

```bibtex
@misc{aropd2026,
  title = {Beyond Absolute Imitation: Anchored Residual Guidance for Privileged On-Policy Distillation},
  author = {Wenhao Zhu},
  year = {2026},
  note = {Preprint}
}
```

The citation will be updated after the arXiv identifier is available.

## GitHub Pages Setup

This repository serves the project page from:

```text
main / docs
```

If Pages is not enabled, open repository `Settings -> Pages`, choose `Deploy from a branch`, select branch `main`, and set the folder to `/docs`.

## Security

Do not commit GitHub tokens, API keys, private checkpoints, or private datasets. If a personal access token has been pasted into any chat, document, or terminal output, revoke it and generate a new one.
