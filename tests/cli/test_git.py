"""Tests for git CLI functionality."""

from unittest.mock import Mock, patch

import pytest

from llamabot.cli.git import write_release_notes


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
                            "## Version 0.1.0\n\nInitial release"
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
                            assert "## Version 0.1.0" in expected_file.read_text()

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
                            "## Version 0.2.0\n\nSecond release"
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
                            assert "## Version 0.2.0" in expected_file.read_text()

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
                            "## Version 0.3.0\n\nThird release"
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
                            assert "## Version 0.3.0" in expected_file.read_text()

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
                            "## Version 0.1.0\n\nInitial release"
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
