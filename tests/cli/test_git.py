"""Tests for git CLI functionality."""

import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from llamabot.cli.git import compose, hooks, resolve_git_paths, write_release_notes


def run_git(repo: Path, *args: str) -> None:
    """Run a git command inside ``repo``, raising on failure.

    :param repo: Directory in which to run the command.
    :param args: Git subcommand and its arguments.
    """
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def create_repository(tmp_path: Path) -> Path:
    """Create a temp repository with one commit, ready for worktree tests.

    :param tmp_path: pytest-provided temporary directory.
    :return: Path to the initialized repository root.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    run_git(repo, "init")
    run_git(repo, "config", "user.email", "test@example.com")
    run_git(repo, "config", "user.name", "Test User")
    (repo / "README.md").write_text("# test\n")
    run_git(repo, "add", "README.md")
    run_git(repo, "commit", "-m", "initial commit")
    return repo


class TestResolveGitPaths:
    """Test resolve_git_paths across repository layouts."""

    def test_plain_repo_root(self, tmp_path, monkeypatch):
        """In a plain checkout, git_dir and common_dir both equal .git."""
        repo = create_repository(tmp_path)
        monkeypatch.chdir(repo)
        git_dir, common_dir = resolve_git_paths()
        assert git_dir == repo / ".git"
        assert common_dir == repo / ".git"

    def test_linked_worktree(self, tmp_path, monkeypatch):
        """In a linked worktree, git_dir is per-worktree, common_dir shared."""
        repo = create_repository(tmp_path)
        worktree = tmp_path / "linked-worktree"
        run_git(repo, "worktree", "add", str(worktree))
        monkeypatch.chdir(worktree)
        git_dir, common_dir = resolve_git_paths()
        # GitPython returns common_dir lexically as
        # .../worktrees/<name>/../.., so compare resolved paths.
        assert (
            git_dir.resolve()
            == (repo / ".git" / "worktrees" / "linked-worktree").resolve()
        )
        assert common_dir.resolve() == (repo / ".git").resolve()

    def test_from_subdirectory(self, tmp_path, monkeypatch):
        """From a nested subdirectory, the enclosing repository is found."""
        repo = create_repository(tmp_path)
        subdir = repo / "src" / "deep"
        subdir.mkdir(parents=True)
        monkeypatch.chdir(subdir)
        git_dir, common_dir = resolve_git_paths()
        assert git_dir == repo / ".git"
        assert common_dir == repo / ".git"

    def test_outside_repository_raises_runtimeerror(self, tmp_path, monkeypatch):
        """Outside any git repository, a RuntimeError with guidance is raised."""
        outside = tmp_path / "not-a-repo"
        outside.mkdir()
        monkeypatch.chdir(outside)
        with pytest.raises(RuntimeError, match="inside a git repository"):
            resolve_git_paths()


class TestHooks:
    """Test the hooks command across repository layouts."""

    def test_plain_repo_installs_hook(self, tmp_path, monkeypatch):
        """In a plain checkout, the hook lands in .git/hooks/ (GIT-PATH-006)."""
        repo = create_repository(tmp_path)
        monkeypatch.chdir(repo)
        hooks()
        hook = repo / ".git" / "hooks" / "prepare-commit-msg"
        assert hook.exists()
        assert hook.stat().st_mode & 0o111  # executable

    def test_linked_worktree_installs_hook_in_common_dir(self, tmp_path, monkeypatch):
        """In a worktree, the hook lands in the shared hooks dir (GIT-PATH-003)."""
        repo = create_repository(tmp_path)
        worktree = tmp_path / "linked-worktree"
        run_git(repo, "worktree", "add", str(worktree))
        monkeypatch.chdir(worktree)
        hooks()
        assert (worktree / ".git").is_file()  # precondition: linked worktree
        hook = repo / ".git" / "hooks" / "prepare-commit-msg"
        assert hook.exists()
        assert "llamabot git compose" in hook.read_text()

    def test_from_subdirectory_installs_hook(self, tmp_path, monkeypatch):
        """From a subdirectory, the hook still lands in .git/hooks/."""
        repo = create_repository(tmp_path)
        subdir = repo / "src"
        subdir.mkdir()
        monkeypatch.chdir(subdir)
        hooks()
        assert (repo / ".git" / "hooks" / "prepare-commit-msg").exists()

    def test_outside_repository_raises_runtimeerror(self, tmp_path, monkeypatch):
        """Outside any git repository, a RuntimeError with guidance is raised."""
        outside = tmp_path / "not-a-repo"
        outside.mkdir()
        monkeypatch.chdir(outside)
        with pytest.raises(RuntimeError, match="inside a git repository"):
            hooks()


class TestCompose:
    """Test the compose command across repository layouts."""

    @staticmethod
    def mocked_compose(monkeypatch, message: str) -> None:
        """Patch diff/bot dependencies and run compose with a canned message.

        :param monkeypatch: pytest monkeypatch fixture.
        :param message: Commit message the mocked bot produces.
        """
        monkeypatch.setattr("llamabot.cli.git.get_git_diff", lambda: "fake diff")
        mock_bot = Mock()
        mock_bot.return_value.format.return_value.content = message
        monkeypatch.setattr("llamabot.cli.git.commitbot", lambda model_name: mock_bot)
        compose()

    def test_plain_repo_writes_commit_editmsg(self, tmp_path, monkeypatch):
        """In a plain checkout, the message lands in .git/COMMIT_EDITMSG."""
        repo = create_repository(tmp_path)
        monkeypatch.chdir(repo)
        self.mocked_compose(monkeypatch, "feat: plain checkout")
        assert (repo / ".git" / "COMMIT_EDITMSG").read_text() == "feat: plain checkout"

    def test_linked_worktree_writes_worktree_commit_editmsg(
        self, tmp_path, monkeypatch
    ):
        """In a worktree, the message lands in the worktree admin dir (GIT-PATH-004)."""
        repo = create_repository(tmp_path)
        worktree = tmp_path / "linked-worktree"
        run_git(repo, "worktree", "add", str(worktree))
        monkeypatch.chdir(worktree)
        self.mocked_compose(monkeypatch, "feat: linked worktree")
        assert (worktree / ".git").is_file()  # precondition: linked worktree
        editmsg = repo / ".git" / "worktrees" / "linked-worktree" / "COMMIT_EDITMSG"
        assert editmsg.read_text() == "feat: linked worktree"

    def test_from_subdirectory_writes_commit_editmsg(self, tmp_path, monkeypatch):
        """From a subdirectory, the message still lands in .git/COMMIT_EDITMSG."""
        repo = create_repository(tmp_path)
        subdir = repo / "src"
        subdir.mkdir()
        monkeypatch.chdir(subdir)
        self.mocked_compose(monkeypatch, "feat: from subdir")
        assert (repo / ".git" / "COMMIT_EDITMSG").read_text() == "feat: from subdir"

    def test_outside_repository_raises_runtimeerror(self, tmp_path, monkeypatch):
        """Outside any git repository, a RuntimeError with guidance is raised."""
        outside = tmp_path / "not-a-repo"
        outside.mkdir()
        monkeypatch.chdir(outside)
        with pytest.raises(RuntimeError, match="inside a git repository"):
            compose()


class TestWriteReleaseNotes:
    """Test the write_release_notes function."""

    def test_no_tags_raises_error(self, tmp_path):
        """Test that ValueError is raised when no tags exist."""
        with patch("llamabot.cli.git.here", return_value=str(tmp_path)):
            with patch("git.Repo") as mock_repo_class:
                mock_repo = Mock()
                mock_repo.tags = []
                mock_repo_class.return_value = mock_repo

                with pytest.raises(ValueError, match="No tags found"):
                    write_release_notes()

    def test_one_tag_first_release(self, tmp_path):
        """Test handling of first release with one tag."""
        with patch("llamabot.cli.git.here", return_value=str(tmp_path)):
            with patch("git.Repo") as mock_repo_class:
                # Mock the repository and tags
                mock_repo = Mock()
                mock_tag = Mock()
                mock_tag.name = "v0.1.0"
                mock_tag.commit.hexsha = "abc123"
                mock_tag.commit.committed_datetime = "2023-01-01T00:00:00"
                mock_repo.tags = [mock_tag]
                mock_repo.git.log.return_value = "commit abc123\nInitial commit"
                mock_repo_class.return_value = mock_repo

                # Mock the bot and console
                with patch("llamabot.cli.git.Console"):
                    with patch("llamabot.cli.git.SimpleBot") as mock_bot_class:
                        mock_bot = Mock()
                        mock_bot.return_value.content = (
                            "# Version v0.1.0\n\nInitial release"
                        )
                        mock_bot_class.return_value = mock_bot

                        # Mock the compose_release_notes function
                        with patch(
                            "llamabot.cli.git.compose_release_notes",
                            return_value="prompt",
                        ):
                            write_release_notes(release_notes_dir=tmp_path)

                            # Verify git.log was called with no arguments (all commits)
                            mock_repo.git.log.assert_called_once_with()

                            # Verify the file was written with the correct name
                            expected_file = tmp_path / "v0.1.0.md"
                            assert expected_file.exists()
                            assert "# Version v0.1.0" in expected_file.read_text()

    def test_two_tags_second_release(self, tmp_path):
        """Test handling of second release with two tags."""
        with patch("llamabot.cli.git.here", return_value=str(tmp_path)):
            with patch("git.Repo") as mock_repo_class:
                # Mock the repository and tags
                mock_repo = Mock()
                mock_tag1 = Mock()
                mock_tag1.name = "v0.1.0"
                mock_tag1.commit.hexsha = "abc123"
                mock_tag1.commit.committed_datetime = "2023-01-01T00:00:00"
                mock_tag2 = Mock()
                mock_tag2.name = "v0.2.0"
                mock_tag2.commit.hexsha = "def456"
                mock_tag2.commit.committed_datetime = "2023-01-02T00:00:00"
                mock_repo.tags = [mock_tag1, mock_tag2]
                mock_repo.git.log.return_value = "commit def456\nSecond release"
                mock_repo_class.return_value = mock_repo

                # Mock the bot and console
                with patch("llamabot.cli.git.Console"):
                    with patch("llamabot.cli.git.SimpleBot") as mock_bot_class:
                        mock_bot = Mock()
                        mock_bot.return_value.content = (
                            "# Version v0.2.0\n\nSecond release"
                        )
                        mock_bot_class.return_value = mock_bot

                        # Mock the compose_release_notes function
                        with patch(
                            "llamabot.cli.git.compose_release_notes",
                            return_value="prompt",
                        ):
                            write_release_notes(release_notes_dir=tmp_path)

                            # Verify git.log was called with the correct range
                            mock_repo.git.log.assert_called_once_with("abc123..def456")

                            # Verify the file was written with the newest tag name
                            expected_file = tmp_path / "v0.2.0.md"
                            assert expected_file.exists()
                            assert "# Version v0.2.0" in expected_file.read_text()

    def test_three_plus_tags_subsequent_release(self, tmp_path):
        """Test handling of subsequent releases with three or more tags."""
        with patch("llamabot.cli.git.here", return_value=str(tmp_path)):
            with patch("git.Repo") as mock_repo_class:
                # Mock the repository and tags
                mock_repo = Mock()
                mock_tag1 = Mock()
                mock_tag1.name = "v0.1.0"
                mock_tag1.commit.hexsha = "abc123"
                mock_tag1.commit.committed_datetime = "2023-01-01T00:00:00"
                mock_tag2 = Mock()
                mock_tag2.name = "v0.2.0"
                mock_tag2.commit.hexsha = "def456"
                mock_tag2.commit.committed_datetime = "2023-01-02T00:00:00"
                mock_tag3 = Mock()
                mock_tag3.name = "v0.3.0"
                mock_tag3.commit.hexsha = "ghi789"
                mock_tag3.commit.committed_datetime = "2023-01-03T00:00:00"
                mock_repo.tags = [mock_tag1, mock_tag2, mock_tag3]
                mock_repo.git.log.return_value = "commit ghi789\nThird release"
                mock_repo_class.return_value = mock_repo

                # Mock the bot and console
                with patch("llamabot.cli.git.Console"):
                    with patch("llamabot.cli.git.SimpleBot") as mock_bot_class:
                        mock_bot = Mock()
                        mock_bot.return_value.content = (
                            "# Version v0.3.0\n\nThird release"
                        )
                        mock_bot_class.return_value = mock_bot

                        # Mock the compose_release_notes function
                        with patch(
                            "llamabot.cli.git.compose_release_notes",
                            return_value="prompt",
                        ):
                            write_release_notes(release_notes_dir=tmp_path)

                            # Verify git.log was called with the correct range (last two tags)
                            mock_repo.git.log.assert_called_once_with("def456..ghi789")

                            # Verify the file was written with the newest tag name
                            expected_file = tmp_path / "v0.3.0.md"
                            assert expected_file.exists()
                            assert "# Version v0.3.0" in expected_file.read_text()

    def test_custom_release_notes_dir(self, tmp_path):
        """Test that custom release notes directory is created and used."""
        custom_dir = tmp_path / "custom_releases"

        with patch("llamabot.cli.git.here", return_value=str(tmp_path)):
            with patch("git.Repo") as mock_repo_class:
                # Mock the repository and tags
                mock_repo = Mock()
                mock_tag = Mock()
                mock_tag.name = "v0.1.0"
                mock_tag.commit.hexsha = "abc123"
                mock_tag.commit.committed_datetime = "2023-01-01T00:00:00"
                mock_repo.tags = [mock_tag]
                mock_repo.git.log.return_value = "commit abc123\nInitial commit"
                mock_repo_class.return_value = mock_repo

                # Mock the bot and console
                with patch("llamabot.cli.git.Console"):
                    with patch("llamabot.cli.git.SimpleBot") as mock_bot_class:
                        mock_bot = Mock()
                        mock_bot.return_value.content = (
                            "# Version v0.1.0\n\nInitial release"
                        )
                        mock_bot_class.return_value = mock_bot

                        # Mock the compose_release_notes function
                        with patch(
                            "llamabot.cli.git.compose_release_notes",
                            return_value="prompt",
                        ):
                            write_release_notes(release_notes_dir=custom_dir)

                            # Verify the custom directory was created
                            assert custom_dir.exists()

                            # Verify the file was written in the custom directory
                            expected_file = custom_dir / "v0.1.0.md"
                            assert expected_file.exists()

    def test_git_import_error(self, tmp_path):
        """Test that ImportError is raised when git is not available."""
        with patch("llamabot.cli.git.here", return_value=str(tmp_path)):
            with patch(
                "builtins.__import__", side_effect=ImportError("No module named 'git'")
            ):
                with pytest.raises(ImportError, match="git is not installed"):
                    write_release_notes()

    def test_explicit_version_flag(self, tmp_path):
        """Test that explicit --version flag works correctly."""
        with patch("llamabot.cli.git.here", return_value=str(tmp_path)):
            with patch("git.Repo") as mock_repo_class:
                # Mock repository with two tags
                mock_repo = Mock()
                mock_tag1 = Mock()
                mock_tag1.name = "v0.1.0"
                mock_tag1.commit.hexsha = "abc123"
                mock_tag1.commit.committed_datetime = "2023-01-01T00:00:00"
                mock_tag2 = Mock()
                mock_tag2.name = "v0.2.0"
                mock_tag2.commit.hexsha = "def456"
                mock_tag2.commit.committed_datetime = "2023-01-02T00:00:00"
                mock_repo.tags = [mock_tag1, mock_tag2]
                mock_repo.git.log.return_value = (
                    "commit def456\nBump version: 0.1.0 → 0.2.0"
                )
                mock_repo_class.return_value = mock_repo

                with patch("llamabot.cli.git.Console"):
                    with patch("llamabot.cli.git.SimpleBot") as mock_bot_class:
                        mock_bot = Mock()
                        # LLM should use the explicit version (0.3.0) not what's in commits (0.2.0)
                        mock_bot.return_value.content = (
                            "# Version 0.3.0\n\nThird release"
                        )
                        mock_bot_class.return_value = mock_bot

                        with patch(
                            "llamabot.cli.git.compose_release_notes"
                        ) as mock_compose:
                            mock_compose.return_value = "prompt"

                            # Call with explicit version
                            write_release_notes(
                                release_notes_dir=tmp_path, version="0.3.0"
                            )

                            # Verify compose_release_notes was called with version parameter
                            mock_compose.assert_called_once()
                            args = mock_compose.call_args[0]
                            assert args[1] == "0.3.0"  # Second arg should be version

                            # Verify file was written with explicit version (no 'v' prefix added)
                            expected_file = tmp_path / "0.3.0.md"
                            assert expected_file.exists()
                            assert "# Version 0.3.0" in expected_file.read_text()

    def test_explicit_version_with_v_prefix(self, tmp_path):
        """Test that explicit --version flag respects 'v' prefix as-is."""
        with patch("llamabot.cli.git.here", return_value=str(tmp_path)):
            with patch("git.Repo") as mock_repo_class:
                # Mock repository with one tag
                mock_repo = Mock()
                mock_tag = Mock()
                mock_tag.name = "v0.1.0"
                mock_tag.commit.hexsha = "abc123"
                mock_tag.commit.committed_datetime = "2023-01-01T00:00:00"
                mock_repo.tags = [mock_tag]
                mock_repo.git.log.return_value = "commit abc123\nInitial commit"
                mock_repo_class.return_value = mock_repo

                with patch("llamabot.cli.git.Console"):
                    with patch("llamabot.cli.git.SimpleBot") as mock_bot_class:
                        mock_bot = Mock()
                        mock_bot.return_value.content = (
                            "# Version v0.2.0\n\nSecond release"
                        )
                        mock_bot_class.return_value = mock_bot

                        with patch(
                            "llamabot.cli.git.compose_release_notes"
                        ) as mock_compose:
                            mock_compose.return_value = "prompt"

                            # Call with explicit version that has 'v' prefix
                            write_release_notes(
                                release_notes_dir=tmp_path, version="v0.2.0"
                            )

                            # Verify compose_release_notes was called with version as-is (with 'v' prefix)
                            mock_compose.assert_called_once()
                            args = mock_compose.call_args[0]
                            assert args[1] == "v0.2.0"

                            # Verify file was written with 'v' prefix
                            expected_file = tmp_path / "v0.2.0.md"
                            assert expected_file.exists()

    def test_backward_compatibility_without_version_flag(self, tmp_path):
        """Test that omitting --version maintains backward compatibility."""
        with patch("llamabot.cli.git.here", return_value=str(tmp_path)):
            with patch("git.Repo") as mock_repo_class:
                # Mock repository with three tags (same as test_three_plus_tags_subsequent_release)
                mock_repo = Mock()
                mock_tag1 = Mock()
                mock_tag1.name = "v0.1.0"
                mock_tag1.commit.hexsha = "abc123"
                mock_tag1.commit.committed_datetime = "2023-01-01T00:00:00"
                mock_tag2 = Mock()
                mock_tag2.name = "v0.2.0"
                mock_tag2.commit.hexsha = "def456"
                mock_tag2.commit.committed_datetime = "2023-01-02T00:00:00"
                mock_tag3 = Mock()
                mock_tag3.name = "v0.3.0"
                mock_tag3.commit.hexsha = "ghi789"
                mock_tag3.commit.committed_datetime = "2023-01-03T00:00:00"
                mock_repo.tags = [mock_tag1, mock_tag2, mock_tag3]
                mock_repo.git.log.return_value = "commit ghi789\nThird release"
                mock_repo_class.return_value = mock_repo

                with patch("llamabot.cli.git.Console"):
                    with patch("llamabot.cli.git.SimpleBot") as mock_bot_class:
                        mock_bot = Mock()
                        mock_bot.return_value.content = (
                            "# Version v0.3.0\n\nThird release"
                        )
                        mock_bot_class.return_value = mock_bot

                        with patch(
                            "llamabot.cli.git.compose_release_notes"
                        ) as mock_compose:
                            mock_compose.return_value = "prompt"

                            # Call without explicit version (backward compatibility mode)
                            write_release_notes(release_notes_dir=tmp_path)

                            # Verify git.log was called with the correct range (last two tags)
                            mock_repo.git.log.assert_called_once_with("def456..ghi789")

                            # Verify compose_release_notes was called with version from newest tag
                            mock_compose.assert_called_once()
                            args = mock_compose.call_args[0]
                            assert (
                                args[1] == "v0.3.0"
                            )  # Version from v0.3.0 tag (as-is)

                            # Verify the file was written with the newest tag name
                            expected_file = tmp_path / "v0.3.0.md"
                            assert expected_file.exists()
                            assert "# Version v0.3.0" in expected_file.read_text()

    def test_empty_version_string_falls_back_to_tag(self, tmp_path):
        """Test that empty/whitespace version string falls back to tag inference."""
        with patch("llamabot.cli.git.here", return_value=str(tmp_path)):
            with patch("git.Repo") as mock_repo_class:
                # Mock repository with three tags
                mock_repo = Mock()
                mock_tag1 = Mock()
                mock_tag1.name = "v0.1.0"
                mock_tag1.commit.hexsha = "abc123"
                mock_tag1.commit.committed_datetime = "2023-01-01T00:00:00"
                mock_tag2 = Mock()
                mock_tag2.name = "v0.2.0"
                mock_tag2.commit.hexsha = "def456"
                mock_tag2.commit.committed_datetime = "2023-01-02T00:00:00"
                mock_tag3 = Mock()
                mock_tag3.name = "v0.3.0"
                mock_tag3.commit.hexsha = "ghi789"
                mock_tag3.commit.committed_datetime = "2023-01-03T00:00:00"
                mock_repo.tags = [mock_tag1, mock_tag2, mock_tag3]
                mock_repo.git.log.return_value = "commit ghi789\nThird release"
                mock_repo_class.return_value = mock_repo

                with patch("llamabot.cli.git.Console"):
                    with patch("llamabot.cli.git.SimpleBot") as mock_bot_class:
                        mock_bot = Mock()
                        mock_bot.return_value.content = (
                            "# Version v0.3.0\n\nThird release"
                        )
                        mock_bot_class.return_value = mock_bot

                        with patch(
                            "llamabot.cli.git.compose_release_notes"
                        ) as mock_compose:
                            mock_compose.return_value = "prompt"

                            # Call with whitespace-only version string
                            write_release_notes(
                                release_notes_dir=tmp_path, version="   "
                            )

                            # Verify it fell back to tag inference
                            mock_repo.git.log.assert_called_once_with("def456..ghi789")

                            # Verify compose_release_notes was called with version from newest tag
                            mock_compose.assert_called_once()
                            args = mock_compose.call_args[0]
                            assert (
                                args[1] == "v0.3.0"
                            )  # Version from v0.3.0 tag (as-is), not empty string

                            # Verify the file was written with the tag name
                            expected_file = tmp_path / "v0.3.0.md"
                            assert expected_file.exists()
                            assert "# Version v0.3.0" in expected_file.read_text()
