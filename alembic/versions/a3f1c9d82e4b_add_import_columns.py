"""Add is_imported and source_user columns for remote sync imports

Revision ID: a3f1c9d82e4b
Revises: 8b7e4a5927d7
Create Date: 2026-02-25 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a3f1c9d82e4b"
down_revision: Union[str, Sequence[str], None] = "8b7e4a5927d7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add import tracking columns to notes table."""
    op.add_column(
        "notes",
        sa.Column(
            "is_imported",
            sa.String(5),
            server_default="false",
            nullable=False,
        ),
    )
    op.add_column(
        "notes",
        sa.Column("source_user", sa.String(255), nullable=True),
    )
    op.create_index("ix_notes_is_imported", "notes", ["is_imported"])
    op.create_index("ix_notes_source_user", "notes", ["source_user"])


def downgrade() -> None:
    """Remove import tracking columns from notes table."""
    op.drop_index("ix_notes_source_user")
    op.drop_index("ix_notes_is_imported")
    op.drop_column("notes", "source_user")
    op.drop_column("notes", "is_imported")
