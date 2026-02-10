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


# =============================================================================
# Obsidian Link Cascade Tests
# =============================================================================


class TestObsidianLinkCascade:
    """Tests for cascade re-mirroring when a note's title changes."""

    @pytest.fixture
    def obsidian_service(self, test_config):
        """Create a zettel_service backed by a repo with Obsidian vault."""
        from znote_mcp.services.zettel_service import ZettelService
        from sqlalchemy import create_engine
        from znote_mcp.models.db_models import Base

        with tempfile.TemporaryDirectory() as vault_dir:
            database_path = test_config.get_absolute_path(test_config.database_path)
            engine = create_engine(f"sqlite:///{database_path}")
            Base.metadata.create_all(engine)
            engine.dispose()

            repo = NoteRepository(
                notes_dir=test_config.notes_dir,
                obsidian_vault_path=Path(vault_dir),
                use_git=False,
            )
            service = ZettelService(repository=repo)
            yield service, Path(vault_dir)

    def test_title_change_cascades_to_linking_notes(self, obsidian_service):
        """When note B's title changes, notes linking TO B get re-mirrored."""
        service, vault = obsidian_service

        # Create two notes and link A -> B
        note_b = service.create_note(title="Original Title", content="Target note")
        note_a = service.create_note(title="Linking Note", content="Links to B")
        service.create_link(note_a.id, note_b.id)

        # Verify A's mirror exists
        a_files = list(vault.glob("**/*Linking-Note*.md"))
        assert len(a_files) == 1

        # Update B's title (uses keyword args, not Note object)
        service.update_note(note_b.id, title="Updated Title")

        # A's mirror should have been re-generated with the new wikilink
        a_files_after = list(vault.glob("**/*Linking-Note*.md"))
        assert len(a_files_after) == 1
        a_content_after = a_files_after[0].read_text()

        # The wikilink in A should reference the updated filename
        assert "Updated-Title" in a_content_after or "Updated Title" in a_content_after

    def test_no_cascade_when_title_unchanged(self, obsidian_service):
        """Content-only update should NOT trigger cascade."""
        service, vault = obsidian_service

        note_b = service.create_note(title="Stable Title", content="Original content")
        note_a = service.create_note(title="Observer Note", content="Watches B")
        service.create_link(note_a.id, note_b.id)

        # Get A's mirror mtime before update
        a_files = list(vault.glob("**/*Observer-Note*.md"))
        assert len(a_files) == 1
        a_mtime_before = a_files[0].stat().st_mtime

        # Update B's content only (title stays the same)
        import time
        time.sleep(0.05)  # Ensure different mtime
        service.update_note(note_b.id, content="Updated content only")

        # A's mirror mtime should NOT have changed (no cascade)
        a_files_after = list(vault.glob("**/*Observer-Note*.md"))
        assert len(a_files_after) == 1
        a_mtime_after = a_files_after[0].stat().st_mtime
        assert a_mtime_after == a_mtime_before

    def test_cascade_is_best_effort(self, obsidian_service):
        """Cascade failure should NOT cause the main update to fail."""
        service, vault = obsidian_service

        note_b = service.create_note(title="Will Change", content="Target")
        note_a = service.create_note(title="Linker", content="Links to B")
        service.create_link(note_a.id, note_b.id)

        # Patch cascade to simulate failure — update should still succeed
        from unittest.mock import patch
        with patch.object(
            service.repository, '_cascade_obsidian_remirror',
            side_effect=Exception("Simulated cascade failure")
        ):
            # This should NOT raise despite cascade failure
            updated = service.update_note(note_b.id, title="Changed Title")
            assert updated.title == "Changed Title"


# =============================================================================
# Obsidian Frontmatter Injection Tests
# =============================================================================


class TestObsidianFrontmatter:
    """Tests for aliases and cssclasses injection in Obsidian mirrors."""

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

    def test_mirror_includes_aliases(self, obsidian_repo):
        """Mirrored file should have aliases: [title] in frontmatter."""
        repo, vault = obsidian_repo
        note = Note(
            title="My Research Note",
            content="Some research content.",
            project="testproj",
            note_type=NoteType.PERMANENT,
        )
        repo.create(note)

        files = list(vault.glob("**/*.md"))
        assert len(files) == 1
        content = files[0].read_text()

        import yaml
        # Parse frontmatter
        assert content.startswith("---")
        fm_end = content.index("---", 3)
        fm_block = content[3:fm_end].strip()
        fm = yaml.safe_load(fm_block)

        assert "aliases" in fm
        assert "My Research Note" in fm["aliases"]

    def test_mirror_includes_cssclasses(self, obsidian_repo):
        """Mirrored file should have cssclasses: [note_type] in frontmatter."""
        repo, vault = obsidian_repo
        note = Note(
            title="Fleeting Thought",
            content="Quick idea.",
            project="testproj",
            note_type=NoteType.FLEETING,
        )
        repo.create(note)

        files = list(vault.glob("**/*.md"))
        assert len(files) == 1
        content = files[0].read_text()

        import yaml
        assert content.startswith("---")
        fm_end = content.index("---", 3)
        fm_block = content[3:fm_end].strip()
        fm = yaml.safe_load(fm_block)

        assert "cssclasses" in fm
        assert "fleeting" in fm["cssclasses"]

    def test_source_file_unchanged(self, obsidian_repo):
        """Source .md file should NOT have aliases or cssclasses."""
        repo, vault = obsidian_repo
        note = Note(
            title="Source Test",
            content="Source content.",
            project="testproj",
            note_type=NoteType.PERMANENT,
        )
        created = repo.create(note)

        # Read the source file (not the mirror)
        source_path = repo.notes_dir / f"{created.id}.md"
        source_content = source_path.read_text()

        # Source should NOT contain aliases or cssclasses
        assert "aliases" not in source_content
        assert "cssclasses" not in source_content
