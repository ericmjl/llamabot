# Release pipeline (PR-based, ruleset-compatible) — Low-Level Design

**Created**: 2026-08-16

**Status**: Implemented (PR [#396](https://github.com/ericmjl/llamabot/pull/396)); this document records the design retroactively.

**HLD Link**: [../../high-level-design.md](../../high-level-design.md)

## Requirements

This feature has no separate EARS file; the release workflow's behavior is exercised end-to-end in CI on every release, and its dry run runs on every PR (`(Dry Run) Publish Python Package to PyPI` check).

## Context

The repository ruleset **"Require CI checks on main"** (required status checks + deletion protection on `refs/heads/main`) blocks all direct pushes to `main`, including those from the release workflow. GitHub does not allow the built-in `github-actions[bot]` integration as a ruleset bypass actor (API rejects it: "must be part of the ruleset source or owner organization"). Tags are **not** covered by the ruleset.

Before this design, the release workflow pushed the version bump and tag directly to `main`. The 2026-08-16 release of 0.19.13 was rejected by the ruleset, but the job's `bash -l {0}` default shell does not fail fast, so the rejected push was swallowed and the GitHub release + PyPI publish proceeded. PyPI carried 0.19.13 while `main` stayed at 0.19.12, and every subsequent auto-release attempt died at the "version already exists on PyPI" pre-flight check.

## Design

Ordering remains most-reversible-first; the mutation stage changed from a direct push to a PR flow:

1. **Build & verify** (unchanged) — no external side effects.
2. **Pre-flight** (unchanged) — the target version must not exist on PyPI.
3. **Release PR** — push a `release/vX` branch carrying the two commits (`Bump version to X`, `Add release notes for X`), explicitly dispatch `pr-tests.yaml` on that branch, open a PR, and enable auto-merge with `--rebase`.
4. **Tag** — after the PR merges, tag the post-merge `origin/main` HEAD (the rebased notes commit) and push the tag; tag refs are not ruleset-blocked.
5. **GitHub release** — reversible via `gh release delete`.
6. **PyPI publish** (unchanged) — the only irreversible step, kept last.

Key invariants and the mechanisms that enforce them:

| Invariant | Mechanism |
| --- | --- |
| A rejected push/merge can never be swallowed again | Every mutation step sets `set -euo pipefail` (the `bash -l {0}` default shell does not). |
| PyPI is touched only when `main` actually carries the release | A verification gate re-fetches `origin/main`, asserts its `pyproject.toml` version equals the release version, and asserts the tag exists remotely — before the GitHub release and PyPI steps. |
| Required checks report on the release PR | `pr-tests.yaml` is dispatched explicitly on the release branch, because pushes made with `GITHUB_TOKEN` do not trigger workflow runs (same pattern as `update-ollama-models`). |
| The tag points at the notes commit that is `main` HEAD | The tag step asserts `origin/main` HEAD subject is exactly `Add release notes for X` before tagging; rebase merges preserve commit subjects verbatim. |
| Bot maintenance does not churn releases | The `workflow_run` trigger skips auto-release when the triggering commit subject starts with `Bump version to`, `Add release notes for`, `chore(deps):`, or `chore: update Ollama models list`. |

## Rollback semantics

Force-pushes to `main` are ruleset-blocked, so rollback never attempts them. If a failure happens **after** the release PR merged, the version bump stays on `main` and that version number is skipped by the next release (acceptable: PyPI was not touched). If the failure happens before the merge, the release PR is closed and its branch deleted.

## Known residual state

- The `v0.19.13` tag points at `f9caad8a` (the commit the original run tagged) rather than a commit carrying the 0.19.13 bump. Cosmetic; a manual retag is optional.
- After a post-merge failure, the stranded version number is consumed on `main` but never reaches PyPI. The release notes file for it remains in `docs/releases/`.
