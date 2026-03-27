# Contributing

## Workflow

1. Branch off `main`
2. Open a PR when ready
3. Get 1 review before merging
4. Squash merge (repo enforces this)

Branch names — whatever's descriptive: `add-bias-detection`, `fix-tokenizer-bug`, `refactor-eval-pipeline`, etc.

## Rules for `main`

- Every non-trivial change goes through a PR, even small ones like bug fixes. Trivial follow-ups (typo, missing import) can go directly to `main` with a clear commit message.
- No force pushes to `main`. If you need to fix something after merging, open a PR or push a follow-up commit — don't amend or rebase.

## PRs

Write a brief title and description. The description should explain what changed and why if needed — a couple sentences is fine. Keep both the title and description clear and concise, as they become the squash commit message.

## Misc

After `pip install -r requirements-dev.txt`, run `pre-commit install`. Pre-commit hooks then run automatically on every commit — they format your code with Black and sort imports with isort. If files get reformatted, just re-stage them and commit again.

Flake8 runs in CI on PRs. If it flags something, fix it before merging.
