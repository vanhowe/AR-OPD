# arXiv Submission Checklist

## Account and Identity

- Use the name you want associated with the paper long-term.
- If you do not currently have a university email, a personal email is acceptable for account contact, but an institutional email is usually preferable when available.
- Use a work email only if the work is legitimately connected to that organization and you are comfortable with that address being part of the submission workflow.
- First-time arXiv submitters may need endorsement depending on category and account history.

## Source Package

Prepare a clean arXiv source folder containing:

- main TeX file
- bibliography file or generated `.bbl`
- all custom `.sty` / `.tex` macro files used by the paper
- all figures referenced by the TeX source
- no large logs, build artifacts, private notes, raw datasets, checkpoints, or absolute local paths

Before uploading:

- Compile from the clean source folder.
- Confirm all figures appear.
- Confirm there are no unresolved references or citations.
- Confirm the title, author names, abstract, and acknowledgements are final.
- Confirm the PDF produced by arXiv matches the local PDF closely.

## Metadata

Prepare:

- title
- abstract
- author list and affiliations
- primary category, likely `cs.LG` or `cs.CL` depending on framing
- optional secondary categories, e.g. `cs.AI`
- project page URL
- GitHub code URL
- comments field, e.g. number of pages and figures

## Project Page

Before submitting:

- Publish the GitHub Pages site.
- Add the project page link to the arXiv metadata if desired.
- Add the arXiv link back to the project page after the arXiv identifier is assigned.

## Token Hygiene

If a GitHub personal access token was pasted into any chat, terminal, document, or issue, revoke it immediately and generate a new token. Do not reuse exposed tokens.
