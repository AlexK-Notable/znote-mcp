#!/usr/bin/env python3
"""Migrate existing notes to have appropriate project assignments based on tags."""

import sys
from pathlib import Path
import re
import yaml

# Project detection rules: (project_name, tag_patterns)
# Order matters - first match wins
PROJECT_RULES = [
    # Specific projects first
    ("obsidian-ai-tagger", ["ai-tagger-universe", "obsidian-ai-tagger", "settings-ui-v2", "settings-ui"]),
    ("tagitall", ["tagitall"]),
    ("in-memoria", ["in-memoria", "wave-1", "wave-2", "wave-3", "wave-4"]),
    ("hyprtasking", ["hyprtasking", "hyprtasking-debug"]),
    ("hyprbind", ["hyprbind"]),
    ("variety", ["variety", "smart-selection", "wallust"]),
    ("orcaslicer", ["orcaslicer", "cef", "wxwebview"]),
    ("ignis", ["ignis", "layer-shell"]),
    # Generic categories last
    ("mcp-tools", ["mcp-tools", "mcp-server"]),
]

def detect_project(tags: list[str]) -> str:
    """Detect project from tags."""
    tags_lower = [t.lower() for t in tags]

    for project_name, patterns in PROJECT_RULES:
        for pattern in patterns:
            if any(pattern in tag for tag in tags_lower):
                return project_name

    return "general"

def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from note content."""
    if not content.startswith("---"):
        return {}, content

    # Find end of frontmatter
    end_match = re.search(r'\n---\s*\n', content[3:])
    if not end_match:
        return {}, content

    frontmatter_end = end_match.end() + 3
    frontmatter_yaml = content[4:frontmatter_end - 5]  # Between first --- and second ---
    body = content[frontmatter_end:]

    try:
        frontmatter = yaml.safe_load(frontmatter_yaml) or {}
    except yaml.YAMLError:
        return {}, content

    return frontmatter, body

def rebuild_note(frontmatter: dict, body: str) -> str:
    """Rebuild note with updated frontmatter."""
    # Ensure consistent ordering
    ordered_keys = ['created', 'id', 'project', 'tags', 'title', 'type', 'updated']

    lines = ["---"]
    for key in ordered_keys:
        if key in frontmatter:
            value = frontmatter[key]
            if key == 'tags' and isinstance(value, list):
                lines.append(f"{key}:")
                for tag in value:
                    lines.append(f"- {tag}")
            elif isinstance(value, str) and (':' in value or value.startswith("'")):
                lines.append(f"{key}: '{value}'")
            else:
                lines.append(f"{key}: {value}")

    # Add any remaining keys not in ordered list
    for key, value in frontmatter.items():
        if key not in ordered_keys:
            if isinstance(value, list):
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"- {item}")
            else:
                lines.append(f"{key}: {value}")

    lines.append("---")

    return "\n".join(lines) + "\n" + body

def migrate_notes(notes_dir: Path, dry_run: bool = False) -> dict[str, list[str]]:
    """Migrate all notes to have project fields."""
    results = {}

    for note_file in notes_dir.glob("*.md"):
        content = note_file.read_text()
        frontmatter, body = parse_frontmatter(content)

        if not frontmatter:
            print(f"  SKIP (no frontmatter): {note_file.name}")
            continue

        # Get tags
        tags = frontmatter.get('tags', [])
        if not isinstance(tags, list):
            tags = [tags] if tags else []

        # Detect project
        project = detect_project(tags)

        # Check if already has project
        existing_project = frontmatter.get('project', None)
        if existing_project and existing_project != 'general':
            # Keep existing non-general project
            project = existing_project

        # Track results
        if project not in results:
            results[project] = []
        results[project].append(frontmatter.get('title', note_file.stem))

        # Update frontmatter
        frontmatter['project'] = project

        if not dry_run:
            # Rebuild and save
            new_content = rebuild_note(frontmatter, body)
            note_file.write_text(new_content)

    return results

def main():
    dry_run = "--dry-run" in sys.argv

    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from znote_mcp.config import config

    notes_dir = config.notes_dir
    print(f"Notes directory: {notes_dir}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print()

    results = migrate_notes(notes_dir, dry_run=dry_run)

    # Print summary
    print("\n" + "=" * 60)
    print("PROJECT ASSIGNMENTS")
    print("=" * 60)

    for project in sorted(results.keys()):
        notes = results[project]
        print(f"\n{project} ({len(notes)} notes):")
        for title in sorted(notes)[:5]:  # Show first 5
            print(f"  - {title[:60]}...")
        if len(notes) > 5:
            print(f"  ... and {len(notes) - 5} more")

    total = sum(len(notes) for notes in results.values())
    print(f"\n{'=' * 60}")
    print(f"Total: {total} notes across {len(results)} projects")

    if dry_run:
        print("\nThis was a dry run. Run without --dry-run to apply changes.")

if __name__ == "__main__":
    main()
