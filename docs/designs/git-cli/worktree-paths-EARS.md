# Git CLI worktree paths — EARS

**Parent LLD**: [./LLD.md](./LLD.md)

**Status**: Proposed.

## Repository discovery

- [ ] **GIT-PATH-001**: When the user invokes `llamabot git hooks` or `llamabot git compose` from any subdirectory of a git work tree, the system shall resolve the enclosing repository's git directories without requiring the user to be at the repository root.

- [ ] **GIT-PATH-002**: When the user invokes either command outside a git work tree, the system shall raise a `RuntimeError` whose message instructs the user to run it inside a git repository.

## Linked worktrees

- [ ] **GIT-PATH-003**: When `llamabot git hooks` runs inside a linked worktree, the system shall install `prepare-commit-msg` into the shared hooks directory of the common git dir, such that the hook is active for the main checkout and all linked worktrees.

- [ ] **GIT-PATH-004**: When `llamabot git compose` runs inside a linked worktree, the system shall write the composed commit message to the `COMMIT_EDITMSG` file of the worktree-local admin directory.

- [ ] **GIT-PATH-005**: While `.git` is a file rather than a directory (the linked-worktree marker), the system shall complete both commands without raising `NotADirectoryError`.

## Plain checkouts (regression)

- [ ] **GIT-PATH-006**: When either command runs in a plain (non-worktree) repository root, the system shall behave identically to the previous release: hook at `.git/hooks/prepare-commit-msg`, commit message at `.git/COMMIT_EDITMSG`.
