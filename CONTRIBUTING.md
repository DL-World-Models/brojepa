# Contributing

## Workflow

1. Branch off `main`
2. Open a PR when ready
3. Get 1 review before merging
4. Squash merge (repo enforces this)

Branch names — whatever's descriptive: `add-regularization`, `fix-dataloader-bug`, `refactor-pipeline`, etc.

## Rules for `main`

- Every non-trivial change goes through a PR, even small ones like bug fixes. Trivial follow-ups (typo, missing import) can go directly to `main` with a clear commit message.
- No force pushes to `main`. If you need to fix something after merging, open a PR or push a follow-up commit — don't amend or rebase.

## PRs

Write a brief title and description. The description should explain what changed and why if needed — a couple sentences is fine. Keep both the title and description clear and concise, as they become the squash commit message.

## Dev setup

Create a venv, install dependencies, and enable the git hooks:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
```

If you prefer `uv`, that's fine too.

Pre-commit hooks run automatically on every commit. `ruff` handles both formatting and linting:

- `ruff format` reformats your code.
- `ruff check --fix --exit-zero` auto-fixes safe lint issues and prints any remaining warnings without blocking the commit.

If files get rewritten by either step, just re-stage and commit again.

CI mirrors this exactly (same pinned `ruff==0.15.10`): formatting drift fails the build, lint runs as a non-blocking signal — warnings show up as a yellow indicator on the PR, not a merge block.

## Optional: type checking

We use [basedpyright](https://github.com/detachhead/basedpyright) for editor-side type diagnostics. It's included in `requirements-dev.txt` and is **not** run in pre-commit or CI — it's a signal to help you catch bugs early, not a gate.

The repo ships a `[tool.pyright]` block in `pyproject.toml` that everyone picks up automatically. Install the basedpyright extension in your editor if you want inline diagnostics.
