#!/usr/bin/env python3
"""Sync all zettelkasten notes to the configured Obsidian vault."""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from znote_mcp.config import config
from znote_mcp.storage.note_repository import NoteRepository


def main():
    """Sync notes to Obsidian vault."""
    vault_path = config.get_obsidian_vault_path()

    if not vault_path:
        print("Error: ZETTELKASTEN_OBSIDIAN_VAULT not configured.")
        print("Set it in your .env file:")
        print("  ZETTELKASTEN_OBSIDIAN_VAULT=/path/to/obsidian/vault/zettelkasten")
        sys.exit(1)

    print(f"Syncing notes to: {vault_path}")
    print(f"Source: {config.get_absolute_path(config.notes_dir)}")

    repo = NoteRepository()

    try:
        count = repo.sync_to_obsidian()
        print(f"Successfully synced {count} notes to Obsidian vault.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
