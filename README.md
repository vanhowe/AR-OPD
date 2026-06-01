# AR-OPD Project Page

This folder contains a GitHub Pages-ready project website for:

**Beyond Absolute Imitation: Anchored Residual Guidance for Privileged On-Policy Distillation**

## Files

- `docs/index.html`: project homepage
- `docs/styles.css`: page styling
- `docs/assets/architecture.png`: method architecture figure
- `docs/assets/teacher_reliability.png`: target-reliability diagnostic figure
- `docs/assets/training_dynamics.png`: training dynamics / shortcut / long-rollout figure
- `docs/assets/aropd-paper.pdf`: current paper PDF

## Recommended GitHub Setup

Recommended repo name:

```text
AR-OPD
```

Recommended description:

```text
Project page and resources for AR-OPD: Anchored Residual Guidance for Privileged On-Policy Distillation.
```

After creating the repo, copy this folder's contents into the repo root and enable GitHub Pages:

1. Open the repository settings.
2. Go to `Pages`.
3. Set source to `Deploy from a branch`.
4. Select branch `main` and folder `/docs`.
5. Save.

The project page URL will usually be:

```text
https://vanhowe.github.io/AR-OPD/
```

If you prefer to use the existing `GD-Train-Collab` repository, copy only the `docs/` folder into that repo and enable Pages from `/docs`. The URL will usually be:

```text
https://vanhowe.github.io/GD-Train-Collab/
```

## After arXiv Is Live

Update these placeholders in `docs/index.html`:

- Replace `arXiv coming soon` with the arXiv URL.
- Replace the BibTeX `note = {Preprint}` block with the official arXiv BibTeX.
- Keep the local PDF only if it matches the arXiv version; otherwise link directly to arXiv.

## Security Note

Do not commit GitHub tokens, API keys, private checkpoints, or private datasets to the repository. If a token has been pasted into a chat or document, revoke it and create a new one.
