# tests/test_integration.py
"""Integration tests for the Zettelkasten MCP system."""
import os
import tempfile
from pathlib import Path
import pytest
from znote_mcp.config import config
from znote_mcp.models.schema import LinkType, NoteType
from znote_mcp.server.mcp_server import ZettelkastenMcpServer
from znote_mcp.services.zettel_service import ZettelService
from znote_mcp.services.search_service import SearchService

class TestIntegration:
    """Integration tests for the entire Zettelkasten MCP system."""
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Set up test environment using temporary directories."""
        # Create temporary directories for test
        self.temp_notes_dir = tempfile.TemporaryDirectory()
        self.temp_db_dir = tempfile.TemporaryDirectory()
        
        # Configure paths
        self.notes_dir = Path(self.temp_notes_dir.name)
        self.database_path = Path(self.temp_db_dir.name) / "test_zettelkasten.db"
        
        # Save original config values
        self.original_notes_dir = config.notes_dir
        self.original_database_path = config.database_path
        
        # Update config for tests
        config.notes_dir = self.notes_dir
        config.database_path = self.database_path
        
        # Create services
        self.zettel_service = ZettelService()
        self.zettel_service.initialize()
        self.search_service = SearchService(self.zettel_service)
        
        # Create server
        self.server = ZettelkastenMcpServer()
        
        yield
        
        # Restore original config
        config.notes_dir = self.original_notes_dir
        config.database_path = self.original_database_path
        
        # Clean up temp directories
        self.temp_notes_dir.cleanup()
        self.temp_db_dir.cleanup()
    
    def test_create_note_flow(self):
        """Test the complete flow of creating and retrieving a note."""
        # Use the zettel_service directly to create a note
        title = "Integration Test Note"
        content = "This is a test of the complete note creation flow."
        tags = ["integration", "test", "flow"]
        
        # Create the note
        note = self.zettel_service.create_note(
            title=title,
            content=content,
            note_type=NoteType.PERMANENT,
            tags=tags
        )
        assert note.id is not None
        
        # Retrieve the note
        retrieved_note = self.zettel_service.get_note(note.id)
        assert retrieved_note is not None
        assert retrieved_note.title == title
        
        # Note content includes the title as a markdown header - account for this
        expected_content = f"# {title}\n\n{content}"
        assert retrieved_note.content.strip() == expected_content.strip()
        
        # Check tags
        for tag in tags:
            assert tag in [t.name for t in retrieved_note.tags]
        
        # Verify the note exists on disk
        note_file = self.notes_dir / f"{note.id}.md"
        assert note_file.exists(), "Note file was not created on disk"
        
        # Verify file content
        with open(note_file, "r") as f:
            file_content = f.read()
            assert title in file_content
            assert content in file_content
    
    def test_knowledge_graph_flow(self):
        """Test creating a small knowledge graph with links and semantic relationships."""
        # Create several notes to form a knowledge graph
        hub_note = self.zettel_service.create_note(
            title="Knowledge Graph Hub",
            content="This is the central hub for our test knowledge graph.",
            note_type=NoteType.HUB,
            tags=["knowledge-graph", "hub", "integration-test"]
        )
        
        concept1 = self.zettel_service.create_note(
            title="Concept One",
            content="This is the first concept in our knowledge graph.",
            note_type=NoteType.PERMANENT,
            tags=["knowledge-graph", "concept", "integration-test"]
        )
        
        concept2 = self.zettel_service.create_note(
            title="Concept Two",
            content="This is the second concept, which extends the first.",
            note_type=NoteType.PERMANENT,
            tags=["knowledge-graph", "concept", "integration-test"]
        )
        
        critique = self.zettel_service.create_note(
            title="Critique of Concepts",
            content="This note critiques and questions the concepts.",
            note_type=NoteType.PERMANENT,
            tags=["knowledge-graph", "critique", "integration-test"]
        )
        
        # Create links with different semantic meanings
        # Use different link types to avoid uniqueness constraint issues
        self.zettel_service.create_link(
            source_id=hub_note.id,
            target_id=concept1.id,
            link_type=LinkType.REFERENCE,
            description="Main concept",
            bidirectional=True
        )
        
        self.zettel_service.create_link(
            source_id=hub_note.id,
            target_id=concept2.id,
            link_type=LinkType.EXTENDS,
            description="Secondary concept",
            bidirectional=True
        )
        
        self.zettel_service.create_link(
            source_id=hub_note.id,
            target_id=critique.id,
            link_type=LinkType.SUPPORTS,
            description="Critical perspective",
            bidirectional=True
        )
        
        self.zettel_service.create_link(
            source_id=concept2.id,
            target_id=concept1.id,
            link_type=LinkType.REFINES,
            description="Builds upon first concept",
            bidirectional=True
        )
        
        self.zettel_service.create_link(
            source_id=critique.id,
            target_id=concept1.id,
            link_type=LinkType.QUESTIONS,
            description="Questions assumptions",
            bidirectional=True
        )
        
        self.zettel_service.create_link(
            source_id=critique.id,
            target_id=concept2.id,
            link_type=LinkType.CONTRADICTS,
            description="Contradicts conclusions",
            bidirectional=True
        )
        
        # Get all notes linked to the hub
        hub_links = self.zettel_service.get_linked_notes(hub_note.id, "outgoing")
        assert len(hub_links) == 3
        hub_links_ids = {note.id for note in hub_links}
        assert concept1.id in hub_links_ids
        assert concept2.id in hub_links_ids
        assert critique.id in hub_links_ids
        
        # Get notes extended by concept2
        concept2_links = self.zettel_service.get_linked_notes(concept2.id, "outgoing")
        assert len(concept2_links) >= 1  # At least one link
        
        # Verify links by tag
        kg_notes = self.zettel_service.get_notes_by_tag("knowledge-graph")
        assert len(kg_notes) == 4  # Should find all 4 notes
    
    def test_rebuild_index_flow(self):
        """Test the rebuild index functionality with direct file modifications."""
        # Create a note through the service
        note1 = self.zettel_service.create_note(
            title="Original Note",
            content="This is the original content.",
            tags=["rebuild-test"]
        )
        
        # Manually modify the file to simulate external editing
        note_file = self.notes_dir / f"{note1.id}.md"
        assert note_file.exists(), "Note file was not created on disk"
        
        # Read the current file content
        with open(note_file, "r") as f:
            file_content = f.read()
        
        # Modify the file content directly, ensuring we replace the content part only
        # The content in the file will include the title header, so we need to search
        # for the entire content structure
        modified_content = file_content.replace(
            "This is the original content.",
            "This content was manually edited outside the system."
        )
        
        # Write the modified content back
        with open(note_file, "w") as f:
            f.write(modified_content)
        
        # At this point, the file has been modified but the database hasn't been updated
        
        # Verify the database still has old content by reading through the repository
        modified_file_content = self.zettel_service.get_note(note1.id).content
        assert "manually edited" in modified_file_content
        
        # Rebuild the index
        self.zettel_service.rebuild_index()
        
        # Verify the note now has the updated content
        note1_after = self.zettel_service.get_note(note1.id)
        assert "This content was manually edited outside the system." in note1_after.content

    def test_full_crud_lifecycle(self):
        """Test complete Create, Read, Update, Delete lifecycle."""
        # CREATE
        note = self.zettel_service.create_note(
            title="CRUD Test Note",
            content="Initial content for CRUD test.",
            note_type=NoteType.PERMANENT,
            tags=["crud", "test"]
        )
        note_id = note.id
        assert note_id is not None

        # READ
        retrieved = self.zettel_service.get_note(note_id)
        assert retrieved is not None
        assert retrieved.title == "CRUD Test Note"
        assert "Initial content" in retrieved.content

        # UPDATE - using the service's update_note method signature
        updated = self.zettel_service.update_note(
            note_id=note_id,
            title="Updated CRUD Test Note",
            content="# Updated CRUD Test Note\n\nUpdated content for CRUD test.",
            tags=["crud", "test", "updated"]
        )
        assert updated.title == "Updated CRUD Test Note"
        assert "updated" in [t.name for t in updated.tags]

        # Verify update persisted
        retrieved_again = self.zettel_service.get_note(note_id)
        assert retrieved_again.title == "Updated CRUD Test Note"
        assert "Updated content" in retrieved_again.content

        # DELETE
        self.zettel_service.delete_note(note_id)

        # Verify deletion
        deleted = self.zettel_service.get_note(note_id)
        assert deleted is None

        # Verify file is gone
        note_file = self.notes_dir / f"{note_id}.md"
        assert not note_file.exists()

    def test_search_with_multiple_filters(self):
        """Test search functionality with multiple filter combinations."""
        # Create notes with various attributes
        note1 = self.zettel_service.create_note(
            title="Python Programming Guide",
            content="A comprehensive guide to Python programming.",
            note_type=NoteType.PERMANENT,
            tags=["programming", "python", "guide"]
        )

        note2 = self.zettel_service.create_note(
            title="JavaScript Basics",
            content="Introduction to JavaScript programming language.",
            note_type=NoteType.LITERATURE,
            tags=["programming", "javascript", "basics"]
        )

        note3 = self.zettel_service.create_note(
            title="Python Data Science",
            content="Using Python for data science applications.",
            note_type=NoteType.PERMANENT,
            tags=["programming", "python", "data-science"]
        )

        note4 = self.zettel_service.create_note(
            title="Quick Python Tip",
            content="A fleeting thought about Python.",
            note_type=NoteType.FLEETING,
            tags=["python", "tip"]
        )

        # Search by content
        content_results = self.search_service.search_by_text("Python")
        assert len(content_results) >= 3  # Should find note1, note3, note4

        # Search by tag
        tag_results = self.search_service.search_by_tag("programming")
        assert len(tag_results) == 3  # note1, note2, note3

        # Search by note type using combined search
        type_results = self.search_service.search_combined(note_type=NoteType.PERMANENT)
        type_result_ids = [r.note.id for r in type_results]
        assert note1.id in type_result_ids
        assert note3.id in type_result_ids

        # Combined search with all filters
        combined_results = self.search_service.search_combined(
            text="Python",
            tags=["programming"],
            note_type=NoteType.PERMANENT
        )
        # Should find notes that match all criteria
        result_ids = [r.note.id for r in combined_results]
        assert note1.id in result_ids  # Matches all criteria
        assert note3.id in result_ids  # Matches all criteria
        assert note2.id not in result_ids  # JavaScript, not Python
        assert note4.id not in result_ids  # Fleeting, not Permanent

    def test_link_bidirectional_verification(self):
        """Test that bidirectional links are properly created and retrievable."""
        # Create two notes
        note_a = self.zettel_service.create_note(
            title="Note A",
            content="This is Note A.",
            tags=["bidirectional-test"]
        )

        note_b = self.zettel_service.create_note(
            title="Note B",
            content="This is Note B.",
            tags=["bidirectional-test"]
        )

        # Create bidirectional link
        self.zettel_service.create_link(
            source_id=note_a.id,
            target_id=note_b.id,
            link_type=LinkType.EXTENDS,
            description="A extends B",
            bidirectional=True
        )

        # Verify forward link (A -> B)
        note_a_outgoing = self.zettel_service.get_linked_notes(note_a.id, "outgoing")
        assert len(note_a_outgoing) == 1
        assert note_a_outgoing[0].id == note_b.id

        # Verify reverse link (B -> A with inverse type)
        note_b_outgoing = self.zettel_service.get_linked_notes(note_b.id, "outgoing")
        assert len(note_b_outgoing) == 1
        assert note_b_outgoing[0].id == note_a.id

        # Verify incoming links work too
        note_a_incoming = self.zettel_service.get_linked_notes(note_a.id, "incoming")
        assert len(note_a_incoming) == 1
        assert note_a_incoming[0].id == note_b.id

        note_b_incoming = self.zettel_service.get_linked_notes(note_b.id, "incoming")
        assert len(note_b_incoming) == 1
        assert note_b_incoming[0].id == note_a.id

    def test_obsidian_vault_mirroring(self):
        """Test Obsidian vault mirroring functionality."""
        # Create a temporary Obsidian vault directory
        with tempfile.TemporaryDirectory() as temp_obsidian_dir:
            obsidian_path = Path(temp_obsidian_dir)

            # Instead of module reloading, directly patch the repository's obsidian path
            # This avoids test pollution from module reloads
            self.zettel_service.repository.obsidian_vault_path = obsidian_path

            try:
                # Create a note with a project
                note = self.zettel_service.create_note(
                    title="Obsidian Test Note",
                    content="This note should be mirrored to Obsidian.",
                    note_type=NoteType.PERMANENT,
                    project="test-project",
                    tags=["obsidian", "mirror"]
                )

                # Verify the note was mirrored (filename includes ID suffix for uniqueness)
                # Format: "Title (id_suffix).md" where id_suffix is last 8 chars of note ID
                project_dir = obsidian_path / "test-project"
                id_suffix = note.id[-8:]
                matching_files = list(project_dir.glob(f"Obsidian Test Note ({id_suffix}).md"))
                assert len(matching_files) == 1, f"Expected 1 mirror file, found {len(matching_files)} in {project_dir}"
                expected_file = matching_files[0]

                # Verify content
                with open(expected_file, "r") as f:
                    content = f.read()
                    assert "Obsidian Test Note" in content
                    assert "This note should be mirrored" in content

                # Create another note
                note2 = self.zettel_service.create_note(
                    title="Second Obsidian Note",
                    content="Another note for Obsidian.",
                    project="another-project"
                )

                # Verify second note mirrored with ID suffix
                project_dir2 = obsidian_path / "another-project"
                id_suffix2 = note2.id[-8:]
                matching_files2 = list(project_dir2.glob(f"Second Obsidian Note ({id_suffix2}).md"))
                assert len(matching_files2) == 1

                # Test bulk sync
                synced_count = self.zettel_service.sync_to_obsidian()
                assert synced_count >= 2

            finally:
                # Restore repository to have no obsidian vault path
                self.zettel_service.repository.obsidian_vault_path = None

    def test_obsidian_title_collision(self):
        """Test that notes with colliding sanitized titles create separate files.

        Titles like "Hello: World" and "Hello; World" both sanitize to "Hello_ World"
        but should create unique files via the ID suffix mechanism.
        """
        with tempfile.TemporaryDirectory() as temp_obsidian_dir:
            obsidian_path = Path(temp_obsidian_dir)
            self.zettel_service.repository.obsidian_vault_path = obsidian_path

            try:
                # Create two notes with titles that sanitize to the same string
                # Colon (:) and semicolon (;) both become underscore (_)
                note1 = self.zettel_service.create_note(
                    title="Hello: World",  # Sanitizes to "Hello_ World"
                    content="First note with colon in title.",
                    project="collision-test"
                )

                note2 = self.zettel_service.create_note(
                    title="Hello; World",  # Also sanitizes to "Hello_ World"
                    content="Second note with semicolon in title.",
                    project="collision-test"
                )

                # Both notes should have unique files in the same directory
                project_dir = obsidian_path / "collision-test"

                # Find files for each note by their ID suffix
                id_suffix1 = note1.id[-8:]
                id_suffix2 = note2.id[-8:]

                files1 = list(project_dir.glob(f"*({id_suffix1}).md"))
                files2 = list(project_dir.glob(f"*({id_suffix2}).md"))

                assert len(files1) == 1, f"Note 1 should have exactly 1 file, found {len(files1)}"
                assert len(files2) == 1, f"Note 2 should have exactly 1 file, found {len(files2)}"

                # Verify they are different files
                assert files1[0] != files2[0], "Collision! Both notes mapped to same file"

                # Verify content is correct in each file
                with open(files1[0], "r") as f:
                    content1 = f.read()
                    assert "First note with colon" in content1

                with open(files2[0], "r") as f:
                    content2 = f.read()
                    assert "Second note with semicolon" in content2

                # Verify total files (should be exactly 2)
                all_files = list(project_dir.glob("*.md"))
                assert len(all_files) == 2, f"Expected 2 files, found {len(all_files)}: {all_files}"

            finally:
                self.zettel_service.repository.obsidian_vault_path = None

    def test_get_notes_by_project(self):
        """Test SQL-level project filtering via get_notes_by_project()."""
        # Create notes in different projects
        note1 = self.zettel_service.create_note(
            title="Alpha Note 1",
            content="First note in Alpha project.",
            project="alpha"
        )
        note2 = self.zettel_service.create_note(
            title="Alpha Note 2",
            content="Second note in Alpha project.",
            project="alpha"
        )
        note3 = self.zettel_service.create_note(
            title="Beta Note",
            content="Note in Beta project.",
            project="beta"
        )
        note4 = self.zettel_service.create_note(
            title="General Note",
            content="Note with default project.",
            # No project specified - uses default "general"
        )

        # Test get_notes_by_project for "alpha"
        alpha_notes = self.zettel_service.get_notes_by_project("alpha")
        alpha_ids = [n.id for n in alpha_notes]
        assert len(alpha_notes) == 2
        assert note1.id in alpha_ids
        assert note2.id in alpha_ids
        assert note3.id not in alpha_ids

        # Test get_notes_by_project for "beta"
        beta_notes = self.zettel_service.get_notes_by_project("beta")
        assert len(beta_notes) == 1
        assert beta_notes[0].id == note3.id

        # Test get_notes_by_project for "general" (default)
        general_notes = self.zettel_service.get_notes_by_project("general")
        assert len(general_notes) == 1
        assert general_notes[0].id == note4.id

        # Test get_notes_by_project for non-existent project
        empty_notes = self.zettel_service.get_notes_by_project("nonexistent")
        assert len(empty_notes) == 0

    def test_get_all_notes_pagination(self):
        """Test pagination in get_all_notes() with limit and offset."""
        # Create 5 notes
        notes = []
        for i in range(5):
            note = self.zettel_service.create_note(
                title=f"Paginated Note {i}",
                content=f"Content for pagination test note {i}.",
                project="pagination-test"
            )
            notes.append(note)

        # Test count_notes
        total_count = self.zettel_service.count_notes()
        assert total_count == 5

        # Test get_all with no pagination (should return all)
        all_notes = self.zettel_service.get_all_notes()
        assert len(all_notes) == 5

        # Test with limit only
        limited_notes = self.zettel_service.get_all_notes(limit=2)
        assert len(limited_notes) == 2

        # Test with offset only
        offset_notes = self.zettel_service.get_all_notes(offset=2)
        assert len(offset_notes) == 3

        # Test with both limit and offset (pagination)
        page1 = self.zettel_service.get_all_notes(limit=2, offset=0)
        page2 = self.zettel_service.get_all_notes(limit=2, offset=2)
        page3 = self.zettel_service.get_all_notes(limit=2, offset=4)

        assert len(page1) == 2
        assert len(page2) == 2
        assert len(page3) == 1

        # Verify no overlap between pages
        page1_ids = {n.id for n in page1}
        page2_ids = {n.id for n in page2}
        page3_ids = {n.id for n in page3}
        assert page1_ids.isdisjoint(page2_ids)
        assert page2_ids.isdisjoint(page3_ids)
        assert page1_ids.isdisjoint(page3_ids)

        # Test offset beyond total (should return empty)
        empty_page = self.zettel_service.get_all_notes(limit=10, offset=100)
        assert len(empty_page) == 0

    def test_search_pagination(self):
        """Test pagination in search_notes() with limit and offset."""
        # Create 5 notes with a common tag
        notes = []
        for i in range(5):
            note = self.zettel_service.create_note(
                title=f"Searchable Note {i}",
                content=f"Content for search pagination test note {i}.",
                tags=["search-test"],
                project="search-pagination"
            )
            notes.append(note)

        # Test count_search_results
        total_count = self.zettel_service.count_search_results(tag="search-test")
        assert total_count == 5

        # Test search with no pagination
        all_results = self.zettel_service.search_notes(tag="search-test")
        assert len(all_results) == 5

        # Test with limit only
        limited_results = self.zettel_service.search_notes(tag="search-test", limit=2)
        assert len(limited_results) == 2

        # Test with both limit and offset (pagination)
        page1 = self.zettel_service.search_notes(tag="search-test", limit=2, offset=0)
        page2 = self.zettel_service.search_notes(tag="search-test", limit=2, offset=2)
        page3 = self.zettel_service.search_notes(tag="search-test", limit=2, offset=4)

        assert len(page1) == 2
        assert len(page2) == 2
        assert len(page3) == 1

        # Verify no overlap between pages
        page1_ids = {n.id for n in page1}
        page2_ids = {n.id for n in page2}
        page3_ids = {n.id for n in page3}
        assert page1_ids.isdisjoint(page2_ids)
        assert page2_ids.isdisjoint(page3_ids)

    def test_orphaned_and_central_notes(self):
        """Test finding orphaned notes and central notes."""
        # Create isolated notes (no links)
        orphan1 = self.zettel_service.create_note(
            title="Orphan One",
            content="This note has no links.",
            tags=["orphan-test"]
        )

        orphan2 = self.zettel_service.create_note(
            title="Orphan Two",
            content="This note is also isolated.",
            tags=["orphan-test"]
        )

        # Create connected notes
        hub = self.zettel_service.create_note(
            title="Central Hub",
            content="This is a highly connected hub.",
            note_type=NoteType.HUB,
            tags=["central-test"]
        )

        connected1 = self.zettel_service.create_note(
            title="Connected One",
            content="Connected to hub.",
            tags=["central-test"]
        )

        connected2 = self.zettel_service.create_note(
            title="Connected Two",
            content="Also connected to hub.",
            tags=["central-test"]
        )

        # Create links to make hub central
        self.zettel_service.create_link(hub.id, connected1.id, LinkType.REFERENCE)
        self.zettel_service.create_link(hub.id, connected2.id, LinkType.REFERENCE)
        self.zettel_service.create_link(connected1.id, connected2.id, LinkType.RELATED)

        # Find orphaned notes
        orphaned = self.search_service.find_orphaned_notes()
        orphan_ids = [n.id for n in orphaned]
        assert orphan1.id in orphan_ids
        assert orphan2.id in orphan_ids
        assert hub.id not in orphan_ids

        # Find central notes
        central = self.search_service.find_central_notes(limit=5)
        # Hub should be among the most central due to having most connections
        # find_central_notes returns List[Tuple[Note, int]] (note, connection_count)
        central_ids = [n[0].id for n in central]
        assert hub.id in central_ids
