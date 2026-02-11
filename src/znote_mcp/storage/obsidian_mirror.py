"""Obsidian vault mirroring for Zettelkasten notes.

Mirrors notes to an Obsidian vault with friendly filenames, rewritten
wikilinks, and Obsidian-specific frontmatter (aliases, cssclasses).
Extracted from NoteRepository for cohesion.
"""
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional

import frontmatter

from znote_mcp.models.schema import Note

if TYPE_CHECKING:
    import datetime

logger = logging.getLogger(__name__)


def _sanitize_for_terminal(text: str) -> str:
    """Convert text to a terminal-friendly filename component.

    Replaces spaces and special characters with hyphens/underscores.
    Imported here to avoid circular dependency on note_repository.
    """
    if not text:
        return ""
    result = text.replace(":", " ").replace(";", " ").replace("/", " ").replace("\\", " ")
    words = result.split()
    sanitized_words = []
    for word in words:
        sanitized_word = "".join(
            c if c.isalnum() or c in "-_" else "" for c in word
        )
        if sanitized_word:
            sanitized_words.append(sanitized_word)
    return "-".join(sanitized_words)


class ObsidianMirror:
    """Mirrors Zettelkasten notes to an Obsidian vault directory.

    Args:
        vault_path: Root path of the Obsidian vault.
        note_resolver: Callable that takes a note ID and returns a Note
            or None.  Used to resolve wikilinks during link rewriting.
    """

    def __init__(
        self,
        vault_path: Path,
        note_resolver: Callable[[str], Optional[Note]],
    ) -> None:
        self.vault_path = vault_path
        self._resolve_note = note_resolver

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mirror_note(self, note: Note, markdown: str) -> None:
        """Mirror a single note to the Obsidian vault.

        Creates ``vault/project/purpose/YYYY-MM-DD_Title_id.md``.
        """
        if not self.vault_path:
            return

        safe_project = _sanitize_for_terminal(note.project) or "general"
        purpose_value = note.note_purpose.value if note.note_purpose else "general"
        safe_purpose = _sanitize_for_terminal(purpose_value) or "general"
        safe_filename = self.build_filename(note.title, note.id, note.created_at)

        purpose_dir = self.vault_path / safe_project / safe_purpose
        purpose_dir.mkdir(parents=True, exist_ok=True)

        obsidian_file_path = purpose_dir / f"{safe_filename}.md"

        obsidian_markdown = self.rewrite_links(markdown)
        obsidian_markdown = self.normalize_markdown(obsidian_markdown)
        obsidian_markdown = self.inject_frontmatter(obsidian_markdown, note)

        try:
            with open(obsidian_file_path, "w", encoding="utf-8") as f:
                f.write(obsidian_markdown)
            logger.debug(f"Mirrored note to Obsidian: {obsidian_file_path}")
        except IOError as e:
            logger.warning(f"Failed to mirror note to Obsidian vault: {e}")

    def cascade_remirror(
        self,
        note_id: str,
        incoming_notes: List[Note],
        note_to_markdown: Callable[[Note], str],
    ) -> None:
        """Re-mirror notes that link TO *note_id* (best-effort cascade).

        When a note's title changes, notes linking to it have stale
        wikilinks in their Obsidian mirrors.
        """
        for linked_note in incoming_notes[:50]:
            try:
                md = note_to_markdown(linked_note)
                self.mirror_note(linked_note, md)
            except Exception as inner_err:
                logger.debug(
                    f"Cascade re-mirror failed for {linked_note.id}: {inner_err}"
                )

    def sync_all(
        self,
        notes: List[Note],
        note_to_markdown: Callable[[Note], str],
    ) -> int:
        """Full re-sync: clean old files then mirror all notes.

        Returns:
            Number of notes mirrored.
        """
        self._clean_old_files()
        count = 0
        for note in notes:
            try:
                md = note_to_markdown(note)
                self.mirror_note(note, md)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to mirror note {note.id}: {e}")
        return count

    # ------------------------------------------------------------------
    # Filename building
    # ------------------------------------------------------------------

    @staticmethod
    def build_filename(
        title: str,
        note_id: str,
        created_at: Optional["datetime.datetime"] = None,
    ) -> str:
        """Build a terminal-friendly Obsidian filename.

        Format: ``YYYY-MM-DD_Sanitized-Title_id_suffix``
        """
        safe_title = _sanitize_for_terminal(title)
        id_suffix = note_id[-8:] if len(note_id) >= 8 else note_id

        date_prefix = ""
        if created_at:
            date_prefix = created_at.strftime("%Y-%m-%d") + "_"

        if safe_title:
            return f"{date_prefix}{safe_title}_{id_suffix}"
        return f"{date_prefix}{note_id}"

    # ------------------------------------------------------------------
    # Link rewriting
    # ------------------------------------------------------------------

    def rewrite_links(self, markdown: str) -> str:
        """Rewrite ID-based ``[[id]]`` wikilinks to Obsidian-compatible names."""
        id_pattern = re.compile(r"\[\[(20\d{6}T\d{9,}|20\d{17,})\]\]")

        def _replace(match: re.Match) -> str:
            nid = match.group(1)
            try:
                linked_note = self._resolve_note(nid)
                if linked_note:
                    obsidian_name = self.build_filename(
                        linked_note.title, linked_note.id, linked_note.created_at
                    )
                    return f"[[{obsidian_name}]]"
            except Exception as e:
                logger.debug(f"Could not resolve note {nid} for link rewrite: {e}")
            return match.group(0)

        return id_pattern.sub(_replace, markdown)

    # ------------------------------------------------------------------
    # Markdown normalisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_markdown(markdown: str) -> str:
        """Fix agent-generated markdown tables for Obsidian rendering."""
        lines = markdown.split("\n")
        result: List[str] = []
        i = 0

        while i < len(lines):
            line = lines[i]
            if re.match(r"^\s*\|(.+\|){2,}\s*$", line):
                cells = [c for c in line.split("|") if c.strip()]
                col_count = len(cells)

                result.append(line)
                i += 1

                separator_fragments = []
                while i < len(lines) and re.match(r"^\s*\|[\s\-:]*$", lines[i]):
                    separator_fragments.append(lines[i])
                    i += 1

                if separator_fragments:
                    sep_cell = "------"
                    proper_separator = (
                        "| " + " | ".join([sep_cell] * col_count) + " |"
                    )
                    result.append(proper_separator)
                elif i < len(lines) and re.match(
                    r"^\s*\|[\s\-:]+(\|[\s\-:]+)+\|\s*$", lines[i]
                ):
                    result.append(lines[i])
                    i += 1
            else:
                result.append(line)
                i += 1

        return "\n".join(result)

    @staticmethod
    def inject_frontmatter(markdown: str, note: Note) -> str:
        """Add Obsidian-specific frontmatter (aliases, cssclasses)."""
        try:
            post = frontmatter.loads(markdown)
            post.metadata["aliases"] = [note.title]
            note_type_val = note.note_type.value if note.note_type else "permanent"
            post.metadata["cssclasses"] = [note_type_val]
            return frontmatter.dumps(post)
        except Exception as e:
            logger.warning(f"Failed to inject Obsidian frontmatter: {e}")
            return markdown

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clean_old_files(self) -> None:
        """Remove existing .md mirror files (preserves Obsidian config)."""
        cleaned = 0
        for md_file in self.vault_path.glob("**/*.md"):
            if md_file.name.startswith("."):
                continue
            try:
                md_file.unlink()
                cleaned += 1
            except OSError as e:
                logger.warning(f"Failed to clean old mirror file {md_file}: {e}")
        if cleaned:
            logger.info(f"Cleaned {cleaned} old mirror files before re-sync")
