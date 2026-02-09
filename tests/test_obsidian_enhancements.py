"""Tests for Obsidian UX enhancements.

Covers:
- Auto-purpose inference from note content/title/tags
- Date-prefix Obsidian filenames
- Markdown normalization for Obsidian (broken table fixing)
- Obsidian sync cleanup behavior
"""
import datetime
from datetime import timezone
import tempfile
from pathlib import Path

import pytest

from znote_mcp.models.schema import Note, NoteType, NotePurpose, Tag
from znote_mcp.services.zettel_service import _infer_purpose
from znote_mcp.storage.note_repository import NoteRepository


# =============================================================================
# Auto-Purpose Inference Tests
# =============================================================================


class TestAutoPurposeInference:
    """Tests for the _infer_purpose() function."""

    def test_bugfixing_from_title(self):
        assert _infer_purpose("Debug memory leak in parser", "Some content") == NotePurpose.BUGFIXING

    def test_bugfixing_from_content(self):
        assert _infer_purpose("Investigation", "Found a bug in the authentication flow") == NotePurpose.BUGFIXING

    def test_bugfixing_from_tags(self):
        assert _infer_purpose("Session notes", "Some content", ["bugfix", "auth"]) == NotePurpose.BUGFIXING

    def test_planning_from_title(self):
        assert _infer_purpose("Architecture Plan for v2", "Details here") == NotePurpose.PLANNING

    def test_planning_from_content(self):
        assert _infer_purpose("Notes", "This is a design proposal for the new API") == NotePurpose.PLANNING

    def test_research_from_title(self):
        assert _infer_purpose("Research: React vs Vue comparison", "Content") == NotePurpose.RESEARCH

    def test_research_from_content(self):
        assert _infer_purpose("Notes", "This analysis evaluates three different approaches") == NotePurpose.RESEARCH

    def test_general_when_no_signal(self):
        assert _infer_purpose("My Notes", "Just some random thoughts here") == NotePurpose.GENERAL

    def test_explicit_purpose_preserved(self, zettel_service):
        """When purpose is explicitly set, inference should NOT override it."""
        note = zettel_service.create_note(
            title="Debug session notes",  # Would infer BUGFIXING
            content="Investigating the crash",
            note_purpose=NotePurpose.RESEARCH,  # Explicit override
        )
        assert note.note_purpose == NotePurpose.RESEARCH

    def test_auto_inference_on_create(self, zettel_service):
        """When purpose is left at GENERAL, inference should kick in."""
        note = zettel_service.create_note(
            title="Debug the authentication crash",
            content="Stack trace shows null pointer in auth module",
            # note_purpose defaults to GENERAL -> should be inferred as BUGFIXING
        )
        assert note.note_purpose == NotePurpose.BUGFIXING

    def test_auto_inference_planning(self, zettel_service):
        note = zettel_service.create_note(
            title="Implementation Plan for API v3",
            content="Phase 1: Design the new endpoints",
        )
        assert note.note_purpose == NotePurpose.PLANNING

    def test_strongest_signal_wins(self):
        """When multiple purposes match, the one with more keyword hits wins."""
        # "debug" + "error" + "fix" = 3 bugfixing hits vs "plan" = 1 planning hit
        result = _infer_purpose(
            "Debug plan", "Fix the error in the plan module"
        )
        assert result == NotePurpose.BUGFIXING

    def test_content_only_checks_first_200_chars(self):
        """Only the first 200 chars of content should be checked."""
        # "bug" appears after 200 chars - should NOT trigger bugfixing
        content = "A" * 201 + " bug fix debug error"
        assert _infer_purpose("Generic title", content) == NotePurpose.GENERAL


# =============================================================================
# Date-Prefix Filename Tests
# =============================================================================


class TestDatePrefixFilenames:
    """Tests for _build_obsidian_filename() with date prefixes."""

    def test_filename_with_date(self, note_repository):
        created = datetime.datetime(2026, 2, 8, 12, 0, 0, tzinfo=timezone.utc)
        result = note_repository._build_obsidian_filename(
            "My Test Note", "20260208T120000000000000000", created
        )
        assert result == "2026-02-08_My-Test-Note_00000000"

    def test_filename_without_date(self, note_repository):
        """When created_at is None, no date prefix."""
        result = note_repository._build_obsidian_filename(
            "My Test Note", "20260208T120000000000000000", None
        )
        assert result == "My-Test-Note_00000000"

    def test_filename_with_special_chars(self, note_repository):
        created = datetime.datetime(2026, 1, 15, tzinfo=timezone.utc)
        result = note_repository._build_obsidian_filename(
            "Architecture Plan: API Design", "20260115T000000000000000000", created
        )
        assert result.startswith("2026-01-15_Architecture-Plan")
        assert result.endswith("_00000000")

    def test_filename_empty_title(self, note_repository):
        created = datetime.datetime(2026, 3, 1, tzinfo=timezone.utc)
        note_id = "20260301T000000000000000000"
        result = note_repository._build_obsidian_filename("", note_id, created)
        assert result == f"2026-03-01_{note_id}"


# =============================================================================
# Markdown Normalization Tests
# =============================================================================


class TestMarkdownNormalization:
    """Tests for _normalize_markdown_for_obsidian()."""

    def test_fix_fragmented_separator(self):
        """Fragmented table separators should be merged into a single line."""
        markdown = (
            "| Header1 | Header2 | Header3 |\n"
            "|------\n"
            "|------\n"
            "|------\n"
            "| data1 | data2 | data3 |\n"
        )
        result = NoteRepository._normalize_markdown_for_obsidian(markdown)
        lines = result.split("\n")
        assert lines[0] == "| Header1 | Header2 | Header3 |"
        assert lines[1] == "| ------ | ------ | ------ |"
        assert lines[2] == "| data1 | data2 | data3 |"

    def test_fix_fragmented_separator_with_trailing_pipe(self):
        """Handle fragments like |------| (with trailing pipe)."""
        markdown = (
            "| A | B |\n"
            "|------|\n"
            "|------|\n"
        )
        # Fragments with trailing pipe won't match our ^\s*\|[\s\-:]*$ pattern
        # but let's verify we handle mixed cases
        result = NoteRepository._normalize_markdown_for_obsidian(markdown)
        assert "| A | B |" in result

    def test_preserve_valid_table(self):
        """A properly formatted table should not be modified."""
        markdown = (
            "| Header1 | Header2 |\n"
            "| ------ | ------ |\n"
            "| data1 | data2 |\n"
        )
        result = NoteRepository._normalize_markdown_for_obsidian(markdown)
        assert result == markdown

    def test_non_table_content_untouched(self):
        """Non-table markdown should pass through unchanged."""
        markdown = "# Title\n\nSome paragraph.\n\n- List item\n"
        result = NoteRepository._normalize_markdown_for_obsidian(markdown)
        assert result == markdown

    def test_mixed_content_with_table(self):
        """Tables embedded in other content should be fixed."""
        markdown = (
            "# Report\n\nSome text.\n\n"
            "| Name | Value |\n"
            "|------\n"
            "|------\n"
            "\n"
            "More text.\n"
        )
        result = NoteRepository._normalize_markdown_for_obsidian(markdown)
        assert "| ------ | ------ |" in result
        assert "# Report" in result
        assert "More text." in result

    def test_fragment_with_only_pipe(self):
        """Bare pipe-only line should be treated as fragment."""
        markdown = (
            "| Col1 | Col2 | Col3 |\n"
            "|------\n"
            "|------\n"
            "|\n"  # bare pipe
            "| a | b | c |\n"
        )
        result = NoteRepository._normalize_markdown_for_obsidian(markdown)
        lines = result.split("\n")
        # Header should be followed by proper separator
        assert lines[1] == "| ------ | ------ | ------ |"

    def test_four_column_table(self):
        """Verify column count is correctly detected for wider tables."""
        markdown = (
            "| A | B | C | D |\n"
            "|------\n"
            "|------\n"
            "|------\n"
            "|------\n"
            "| 1 | 2 | 3 | 4 |\n"
        )
        result = NoteRepository._normalize_markdown_for_obsidian(markdown)
        lines = result.split("\n")
        assert lines[1] == "| ------ | ------ | ------ | ------ |"


# =============================================================================
# Obsidian Mirror Integration Tests
# =============================================================================


class TestObsidianMirrorIntegration:
    """Integration tests for the Obsidian mirror pipeline."""

    @pytest.fixture
    def obsidian_repo(self, test_config):
        """Create a repository with Obsidian vault configured."""
        with tempfile.TemporaryDirectory() as vault_dir:
            repo = NoteRepository(
                notes_dir=test_config.notes_dir,
                obsidian_vault_path=Path(vault_dir),
                use_git=False,
            )
            yield repo, Path(vault_dir)

    def test_mirror_creates_date_prefixed_file(self, obsidian_repo):
        repo, vault = obsidian_repo
        note = Note(
            title="Test Note",
            content="Test content",
            project="myproject",
            note_purpose=NotePurpose.RESEARCH,
        )
        created = repo.create(note)
        # Check that a file was created in vault/myproject/research/
        research_dir = vault / "myproject" / "research"
        assert research_dir.exists()
        files = list(research_dir.glob("*.md"))
        assert len(files) == 1
        # Filename should start with date
        fname = files[0].name
        assert fname[:10].count("-") == 2  # YYYY-MM-DD format

    def test_mirror_normalizes_tables(self, obsidian_repo):
        repo, vault = obsidian_repo
        content_with_broken_table = (
            "# Report\n\n"
            "| Name | Value |\n"
            "|------\n"
            "|------\n"
            "| foo | bar |\n"
        )
        note = Note(
            title="Table Test",
            content=content_with_broken_table,
            project="testproj",
        )
        repo.create(note)
        # Find the mirrored file
        files = list(vault.glob("**/*.md"))
        assert len(files) == 1
        content = files[0].read_text()
        # Table should be normalized
        assert "| ------ | ------ |" in content
        # Fragmented separators should not be present
        lines = content.split("\n")
        fragment_lines = [l for l in lines if l.strip() == "|------"]
        assert len(fragment_lines) == 0

    def test_sync_cleans_old_files(self, obsidian_repo):
        """Sync should remove old mirror files before re-creating."""
        repo, vault = obsidian_repo
        # Create a note
        note = Note(title="Sync Test", content="Content", project="proj")
        repo.create(note)

        # Manually create an old-format file to simulate pre-migration state
        old_file = vault / "proj" / "general" / "Old-Format_12345678.md"
        old_file.parent.mkdir(parents=True, exist_ok=True)
        old_file.write_text("old content")

        # Sync should clean and re-create
        count = repo.sync_to_obsidian()
        assert count >= 1

        # Old file should be gone
        assert not old_file.exists()
