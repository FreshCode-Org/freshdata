---
title: First contribution guide
description: A 15-minute, zero-friction walkthrough for making your first pull request to FreshData.
keywords: contribute to freshdata, open source contribution, good first issue, first pull request
---

# First contribution guide

Welcome! You do **not** need to read the entire FreshData architecture or understand the whole decision engine to make your first contribution. 

Follow this 7-step guide to get set up, make a small improvement, and open a clean pull request in under 20 minutes.

---

```text
Choose a task
      │
      ▼
Clone repository
      │
      ▼
Install editable dev environment
      │
      ▼
Run one targeted test
      │
      ▼
Make your small change
      │
      ▼
Run pre-PR checks
      │
      ▼
Open your Pull Request
```

---

## Step 1: Choose a task

Start with something small, self-contained, and reviewable:
* Browse issues labeled [`good first issue`](https://github.com/FreshCode-Org/freshdata/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
* Check the **Level 1 (20–30 min)** section in the [Contributor Roadmap](../community/contributor-roadmap.md).
* Fix a typo, clarify a documentation explanation, or add a missing edge-case unit test.

> **Tip**: Comment on the issue to say you are working on it so nobody duplicates your effort!

---

## Step 2: Fork and clone the repository

Fork the repository on GitHub, then clone your fork locally:

```bash
git clone https://github.com/<your-username>/freshdata.git
cd freshdata
git remote add upstream https://github.com/FreshCode-Org/freshdata.git
```

---

## Step 3: Create a virtual environment and install dependencies

Set up a clean Python $\ge$ 3.9 environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install freshdata in editable mode with development dependencies
pip install -e ".[dev,ml]"
```

---

## Step 4: Run one targeted test

FreshData enforces a strict 93% global test coverage gate in CI. When you run a single test file, pytest will fail the global coverage gate because only a tiny fraction of the library ran.

**The Golden Tip for local development**: Use `--no-cov` while iterating:

```bash
pytest tests/test_simple.py --no-cov
```

You should see 40 tests pass in under 1 second!

---

## Step 5: Make your change

Create a feature branch from `main`:

```bash
git checkout -b fix/my-first-improvement
```

Make your code, documentation, or test change. 

If you added or fixed behavior, add a corresponding test in `tests/`. For example, if you improved date coercion in `src/freshdata/steps/dtypes.py`, add a test function `test_my_specific_date_case()` in `tests/test_dtypes.py`.

Verify your test passes:

```bash
pytest tests/test_dtypes.py -k "test_my_specific_date_case" --no-cov
```

---

## Step 6: Run pre-PR checks

Before opening a pull request, run the same quality checks that CI executes:

```bash
# 1. Check code formatting and linting
ruff check .

# 2. Check static type annotations
mypy src/freshdata

# 3. Run the fast test lane with the CI coverage gate
pytest -m "not online and not large"
```

If `ruff` finds auto-fixable lint issues, you can run `ruff check --fix .`.

---

## Step 7: Commit and open your pull request

Commit your changes with a clear, descriptive message:

```bash
git add .
git commit -m "docs: clarify installation instructions for polars extra"
git push origin fix/my-first-improvement
```

Navigate to [FreshCode-Org/freshdata](https://github.com/FreshCode-Org/freshdata) and click **Compare & pull request**.

Fill out the pull request template:
* **What changed?** A concise summary of your edit.
* **Why?** What problem it solves or the issue it closes (`Fixes #123`).
* **How was it tested?** Mention the commands you ran in Step 6.

A maintainer will review your PR, provide constructive feedback, and guide it through to merge!

---

## Where to get help

* Open a question in [GitHub Discussions](https://github.com/FreshCode-Org/freshdata/discussions) under **Q&A**.
* Read the [Full Architecture Map](https://github.com/FreshCode-Org/freshdata/blob/main/ARCHITECTURE.md) when you are ready to tackle deeper engine components.
