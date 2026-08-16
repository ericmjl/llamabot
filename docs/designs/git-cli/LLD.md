# Git CLI worktree-safe path resolution — Low-Level Design

**Created**: 2026-08-16

**Status**: Implemented — `llamabot/cli/git.py` (`resolve_git_paths`, `hooks`, `compose`); tests in `tests/cli/test_git.py`.

**HLD Link**: [../../high-level-design.md](../../high-level-design.md)

## Requirements (EARS)

Testable requirements for this feature live in:

- [worktree-paths-EARS.md](./worktree-paths-EARS.md) — path resolution inside linked worktrees, subdirectory invocation, non-repository errors.

## Problem

`llamabot/cli/git.py` hardcodes three `.git/...` paths:

| Site | Path | Breakage |
| --- | --- |
| `hooks()` root check (`git.py:192`) | `Path(".git").exists()` | In a linked worktree `.git` is a **file**, so the check passes and the subsequent hook write fails. From a subdirectory the check fails even though the user is inside the repository. |
| `hooks()` hook install (`git.py:199`, `git.py:222`) | `.git/hooks/prepare-commit-msg` | `NotADirectoryError` in worktrees; wrong location from subdirectories. |
| `compose()` message write (`git.py:234`) | `.git/COMMIT_EDITMSG` | `NotADirectoryError` in worktrees; wrong location from subdirectories. |

Observed failure (2026-08-16, release-fix worktree): every `git commit` aborted because the installed `prepare-commit-msg` hook invoked `llamabot git compose`, which raised `[Errno 20] Not a directory: '.git/COMMIT_EDITMSG'`.

## Git's actual layout

A linked worktree has two git directories:

- **Per-worktree admin dir** — the path named by the `.git` file (e.g. `<repo>/.git/worktrees/<name>`). Holds `HEAD`, `index`, `COMMIT_EDITMSG`, `MERGE_MSG`, etc.
- **Common dir** — the main repository's `.git`. Holds `config`, `refs`, and **`hooks/`**.

Git itself resolves such paths with `git rev-parse --git-path <path>`, which knows which locations are per-worktree and which are shared. GitPython (already a dependency used by `get_git_diff`) exposes the same information: `repo.git_dir` (per-worktree) and `repo.common_dir` (shared; equal to `git_dir` in a plain checkout).

## Design

Add one public helper to `llamabot/cli/git.py` (no underscore prefix, per repo convention):

```python
def resolve_git_paths() -> tuple[Path, Path]:
    """Return (git_dir, common_dir) for the enclosing repository.

    :raises RuntimeError: If the current directory is not inside a git work tree.
    """
    from git import Repo

    repo = Repo(search_parent_directories=True)
    return Path(repo.git_dir), Path(repo.common_dir)
```

Callers:

- `hooks()` — replace the `Path(".git").exists()` guard with a call to `resolve_git_paths()` (the `RuntimeError` from `Repo()` when outside a work tree is caught and re-raised with the existing user-facing message). Write the hook to `common_dir / "hooks" / "prepare-commit-msg"` and chmod that path.
- `compose()` — write the composed message to `git_dir / "COMMIT_EDITMSG"`.

Behavior changes, all intentional:

1. Both commands work from any subdirectory of a repository, not only the root. GitPython's `search_parent_directories=True` finds the enclosing work tree.
2. Both commands work inside linked worktrees; the hook lands in the shared hooks directory (so it applies to every worktree, matching `git`'s own hook lookup), and the commit message lands in the worktree-local admin directory where `git commit` expects it.
3. `hooks()`'s error message loses its "must be in the root folder" wording; the new failure condition is simply "not inside a git repository".

Non-goals: `git worktree` management, hook templates beyond `prepare-commit-msg`, and `compose()`'s LLM behavior are unchanged.

## Alternatives considered

- **`git rev-parse --git-path` via subprocess** — equally correct; rejected because the module's sibling code (`get_git_diff`) already standardizes on GitPython and a subprocess adds quoting/process overhead for no gain.
- **Keeping the root-only restriction and documenting it** — rejected: the crash is in an installed hook path, so users in worktrees cannot commit at all until this is fixed.

## Test plan

Tests in `tests/cli/test_git.py`, mirroring existing `unittest.mock` + `typer.testing.CliRunner` patterns; no LLM calls (mock `commitbot`/`SimpleBot` as existing tests do):

1. Real temp repository (initialized with `git init`) — `hooks()` installs the hook under `.git/hooks/`, `compose()` writes `.git/COMMIT_EDITMSG`.
2. Same temp repository plus `git worktree add` — `hooks()` installs the hook under the **common** dir, `compose()` writes the **worktree admin** dir's `COMMIT_EDITMSG`.
3. Subdirectory of the temp repository — both commands succeed without `cd` to the root.
4. Directory outside any repository — both commands fail with the user-facing `RuntimeError` message.
