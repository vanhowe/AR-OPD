# AR-OPD

This repository hosts the project page and public resources for:

**Beyond Absolute Imitation: Anchored Residual Guidance for Privileged On-Policy Distillation**

## Project Page

GitHub Pages source:

```text
docs/
```

Expected project page URL after Pages is enabled:

```text
https://vanhowe.github.io/AR-OPD/
```

To enable the page:

1. Open repository `Settings`.
2. Go to `Pages`.
3. Set source to `Deploy from a branch`.
4. Select branch `main` and folder `/docs`.
5. Save.

## Repository Contents

- `docs/index.html`: project homepage
- `docs/styles.css`: page styling
- `docs/assets/architecture.png`: method architecture figure
- `docs/assets/teacher_reliability.png`: target-reliability diagnostic figure
- `docs/assets/training_dynamics.png`: training dynamics / shortcut / long-rollout figure
- `docs/assets/aropd-paper.pdf`: current paper PDF
- `ARXIV_CHECKLIST.md`: submission checklist and project-page update notes

## After arXiv Is Live

Update these placeholders in `docs/index.html`:

- Replace `arXiv coming soon` with the arXiv URL.
- Replace the BibTeX `note = {Preprint}` block with the official arXiv BibTeX.
- Keep the local PDF only if it matches the arXiv version; otherwise link directly to arXiv.

## Security Note

Do not commit GitHub tokens, API keys, private checkpoints, or private datasets to the repository. If a token has been pasted into a chat or document, revoke it and create a new one.
