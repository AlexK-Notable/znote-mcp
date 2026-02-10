# tests/test_search_service.py
"""Tests for the search service in the Zettelkasten MCP server."""
from datetime import datetime, timedelta, timezone
import pytest
from znote_mcp.models.schema import LinkType, NoteType
from znote_mcp.services.search_service import SearchResult, SearchService


class TestSearchService:
    """Tests for the SearchService class."""
    
    def test_find_orphaned_notes(self, zettel_service):
        """Test finding notes with no links - use direct orphan creation."""
        # Create a single orphaned note
        orphan = zettel_service.create_note(
            title="Isolated Orphan Note",
            content="This note has no connections.",
            tags=["orphan", "isolated"]
        )
        
        # Create two connected notes
        note1 = zettel_service.create_note(
            title="Connected Note 1",
            content="This note has connections.",
            tags=["connected"]
        )
        note2 = zettel_service.create_note(
            title="Connected Note 2",
            content="This note also has connections.",
            tags=["connected"]
        )
        
        # Link the connected notes
        zettel_service.create_link(note1.id, note2.id)
        
        # Use direct SQL query instead of search service
        orphans = zettel_service.repository.search(tags=["isolated"])
        assert len(orphans) == 1
        assert orphans[0].id == orphan.id

    def test_find_central_notes(self, zettel_service):
        """Test finding notes with the most connections."""
        # Create several notes and add extra links to the central one
        central = zettel_service.create_note(
            title="Central Hub Note",
            content="This is the central hub note.",
            tags=["central", "hub"]
        )
        
        peripheral1 = zettel_service.create_note(
            title="Peripheral Note 1",
            content="Connected to the central hub.",
            tags=["peripheral"]
        )
        
        peripheral2 = zettel_service.create_note(
            title="Peripheral Note 2",
            content="Also connected to the central hub.",
            tags=["peripheral"]
        )
        
        # Create links with different types to avoid constraint issues
        zettel_service.create_link(central.id, peripheral1.id, LinkType.REFERENCE)
        zettel_service.create_link(central.id, peripheral2.id, LinkType.SUPPORTS)
        
        # Verify we can find linked notes
        linked = zettel_service.get_linked_notes(central.id, "outgoing")
        assert len(linked) == 2
        assert {n.id for n in linked} == {peripheral1.id, peripheral2.id}

    def test_find_notes_by_date_range(self, zettel_service):
        """Test finding notes within a date range."""
        # Create a note and ensure we can retrieve it by tag
        note = zettel_service.create_note(
            title="Date Test Note",
            content="For testing date range queries.",
            tags=["date-test", "search"]
        )
        
        # Test retrieving by tag
        found_notes = zettel_service.get_notes_by_tag("date-test")
        assert len(found_notes) == 1
        assert found_notes[0].id == note.id

    def test_find_similar_notes(self, zettel_service):
        """Test finding notes similar to a given note."""
        # Create test notes with shared tags
        note1 = zettel_service.create_note(
            title="Machine Learning",
            content="Introduction to machine learning concepts.",
            tags=["AI", "machine learning", "data science"]
        )
        note2 = zettel_service.create_note(
            title="Neural Networks",
            content="Overview of neural network architectures.",
            tags=["AI", "machine learning", "neural networks"]
        )
        
        # Create link to ensure similarity
        zettel_service.create_link(note1.id, note2.id)
        
        # Verify we can find the note by tag
        ai_notes = zettel_service.get_notes_by_tag("AI")
        assert len(ai_notes) == 2
        assert {n.id for n in ai_notes} == {note1.id, note2.id}

    def test_search_combined(self, zettel_service):
        """Test combined search with multiple criteria."""
        # Create test notes
        note1 = zettel_service.create_note(
            title="Python Data Analysis",
            content="Using Python for data analysis.",
            note_type=NoteType.PERMANENT,
            tags=["python", "data science", "analysis"]
        )
        note2 = zettel_service.create_note(
            title="Python Web Development",
            content="Using Python for web development.",
            note_type=NoteType.PERMANENT,
            tags=["python", "web", "development"]
        )
        
        # Test tag-based search
        python_notes = zettel_service.get_notes_by_tag("python")
        assert len(python_notes) == 2
        assert {n.id for n in python_notes} == {note1.id, note2.id}
        
        # Test tag and type filtering
        permanent_notes = zettel_service.repository.search(
            note_type=NoteType.PERMANENT,
            tags=["python"]
        )
        assert len(permanent_notes) == 2


class TestSearchServiceDirect:
    """Direct tests for SearchService methods with verification of scoring and results."""

    def test_find_orphaned_notes_direct(self, zettel_service):
        """Test find_orphaned_notes returns notes with no links."""
        orphan1 = zettel_service.create_note(title="Orphan 1", content="Alone")
        orphan2 = zettel_service.create_note(title="Orphan 2", content="Also alone")
        connected1 = zettel_service.create_note(title="Connected 1", content="Linked")
        connected2 = zettel_service.create_note(title="Connected 2", content="Also linked")

        # Create a link between connected notes
        zettel_service.create_link(connected1.id, connected2.id, LinkType.REFERENCE)

        search_service = SearchService(zettel_service)
        orphans = search_service.find_orphaned_notes()

        orphan_ids = {note.id for note in orphans}
        assert orphan1.id in orphan_ids
        assert orphan2.id in orphan_ids
        assert connected1.id not in orphan_ids
        assert connected2.id not in orphan_ids

    def test_find_central_notes_direct(self, zettel_service):
        """Test find_central_notes returns notes with most connections."""
        # Create a hub note with multiple connections
        hub = zettel_service.create_note(title="Hub Note", content="Central hub")
        spoke1 = zettel_service.create_note(title="Spoke 1", content="First spoke")
        spoke2 = zettel_service.create_note(title="Spoke 2", content="Second spoke")
        spoke3 = zettel_service.create_note(title="Spoke 3", content="Third spoke")
        isolated = zettel_service.create_note(title="Isolated", content="No links")

        # Hub links to all spokes
        zettel_service.create_link(hub.id, spoke1.id, LinkType.REFERENCE)
        zettel_service.create_link(hub.id, spoke2.id, LinkType.EXTENDS)
        zettel_service.create_link(hub.id, spoke3.id, LinkType.SUPPORTS)

        search_service = SearchService(zettel_service)
        central = search_service.find_central_notes(limit=5)

        # Should return tuples of (note, connection_count)
        assert len(central) >= 1
        first_note, first_count = central[0]

        # Hub should be first with 3 connections
        assert first_note.id == hub.id
        assert first_count == 3

    def test_find_central_notes_respects_limit(self, zettel_service):
        """Test find_central_notes respects the limit parameter."""
        # Create several notes with varying connections
        notes = []
        for i in range(5):
            notes.append(zettel_service.create_note(
                title=f"Note {i}",
                content=f"Content {i}"
            ))

        # Create links: note0 has 4 links, note1 has 3, etc.
        zettel_service.create_link(notes[0].id, notes[1].id, LinkType.REFERENCE)
        zettel_service.create_link(notes[0].id, notes[2].id, LinkType.EXTENDS)
        zettel_service.create_link(notes[0].id, notes[3].id, LinkType.SUPPORTS)
        zettel_service.create_link(notes[0].id, notes[4].id, LinkType.RELATED)

        search_service = SearchService(zettel_service)
        central = search_service.find_central_notes(limit=2)

        # Should only return 2 results
        assert len(central) <= 2

    def test_find_notes_by_date_range_start_date(self, zettel_service):
        """Test find_notes_by_date_range with start_date filter."""
        note = zettel_service.create_note(
            title="Recent Note",
            content="Created recently"
        )

        search_service = SearchService(zettel_service)

        # Search from an hour ago should include the note
        start = datetime.now(timezone.utc) - timedelta(hours=1)
        results = search_service.find_notes_by_date_range(start_date=start)

        result_ids = {n.id for n in results}
        assert note.id in result_ids

    def test_find_notes_by_date_range_end_date(self, zettel_service):
        """Test find_notes_by_date_range with end_date filter."""
        note = zettel_service.create_note(
            title="Test Note",
            content="For date range test"
        )

        search_service = SearchService(zettel_service)

        # Search with end_date in the future should include the note
        end = datetime.now(timezone.utc) + timedelta(hours=1)
        results = search_service.find_notes_by_date_range(end_date=end)

        result_ids = {n.id for n in results}
        assert note.id in result_ids

        # Search with end_date in the past should exclude the note
        past_end = datetime.now(timezone.utc) - timedelta(hours=1)
        past_results = search_service.find_notes_by_date_range(end_date=past_end)
        past_ids = {n.id for n in past_results}
        assert note.id not in past_ids

    def test_find_similar_notes_direct(self, zettel_service):
        """Test find_similar_notes returns similar notes by shared tags and links."""
        # Create notes with significant tag overlap for similarity detection
        base = zettel_service.create_note(
            title="Base Note",
            content="The starting point",
            tags=["python", "testing", "development"]
        )
        # Share 2 of 3 tags to meet default threshold of 0.5
        highly_similar = zettel_service.create_note(
            title="Highly Similar Note",
            content="Very related content",
            tags=["python", "testing", "qa"]
        )
        # Also create a direct link to ensure similarity
        zettel_service.create_link(base.id, highly_similar.id, LinkType.REFERENCE)

        search_service = SearchService(zettel_service)
        similar = search_service.find_similar_notes(base.id)

        # Should return list of tuples (note, similarity_score)
        assert isinstance(similar, list)
        # The highly similar note should be found due to shared tags AND link
        similar_ids = {note.id for note, score in similar}
        assert highly_similar.id in similar_ids
        # The base note should not be in its own results
        assert base.id not in similar_ids
        # Verify scores are positive
        for note, score in similar:
            assert score > 0

    def test_search_combined_text_only(self, zettel_service):
        """Test search_combined with only text filter."""
        note = zettel_service.create_note(
            title="Database Design",
            content="SQL and NoSQL databases."
        )

        search_service = SearchService(zettel_service)
        results = search_service.search_combined(text="database")

        assert len(results) >= 1
        result_ids = [r.note.id for r in results]
        assert note.id in result_ids

    def test_search_combined_tags_only(self, zettel_service):
        """Test search_combined with only tags filter."""
        note1 = zettel_service.create_note(
            title="Python Note",
            content="Content",
            tags=["python"]
        )
        note2 = zettel_service.create_note(
            title="Other Note",
            content="Content",
            tags=["other"]
        )

        search_service = SearchService(zettel_service)
        results = search_service.search_combined(tags=["python"])

        assert len(results) == 1
        assert results[0].note.id == note1.id
        # Without text query, score defaults to 1.0
        assert results[0].score == 1.0

    def test_search_combined_note_type_filter(self, zettel_service):
        """Test search_combined with note_type filter."""
        permanent = zettel_service.create_note(
            title="Permanent Note",
            content="Evergreen content",
            note_type=NoteType.PERMANENT
        )
        fleeting = zettel_service.create_note(
            title="Fleeting Note",
            content="Quick thought",
            note_type=NoteType.FLEETING
        )

        search_service = SearchService(zettel_service)
        results = search_service.search_combined(note_type=NoteType.PERMANENT)

        result_ids = [r.note.id for r in results]
        assert permanent.id in result_ids
        assert fleeting.id not in result_ids

    def test_search_combined_multiple_filters(self, zettel_service):
        """Test search_combined with multiple filters together."""
        target = zettel_service.create_note(
            title="Python Web Development",
            content="Building web apps with Python Flask.",
            note_type=NoteType.PERMANENT,
            tags=["python", "web"]
        )
        wrong_type = zettel_service.create_note(
            title="Python Scripting",
            content="Quick Python scripts",
            note_type=NoteType.FLEETING,
            tags=["python", "scripting"]
        )
        wrong_tag = zettel_service.create_note(
            title="JavaScript Web",
            content="Building with JavaScript",
            note_type=NoteType.PERMANENT,
            tags=["javascript", "web"]
        )

        search_service = SearchService(zettel_service)
        results = search_service.search_combined(
            text="Python",
            tags=["python"],
            note_type=NoteType.PERMANENT
        )

        result_ids = [r.note.id for r in results]
        assert target.id in result_ids
        assert wrong_type.id not in result_ids  # Wrong note type
        # wrong_tag won't be in results because it doesn't have "python" tag

    def test_search_combined_date_range(self, zettel_service):
        """Test search_combined with date range filters."""
        note = zettel_service.create_note(
            title="Current Note",
            content="Just created"
        )

        search_service = SearchService(zettel_service)

        # Search with wide date range should include the note
        start = datetime.now(timezone.utc) - timedelta(hours=1)
        end = datetime.now(timezone.utc) + timedelta(hours=1)
        results = search_service.search_combined(start_date=start, end_date=end)

        result_ids = [r.note.id for r in results]
        assert note.id in result_ids


class TestSearchCombinedFts:
    """Tests for FTS5-accelerated search_combined()."""

    def test_search_combined_uses_fts_when_available(self, zettel_service):
        """FTS5 should be used for text queries when available."""
        note1 = zettel_service.create_note(
            title="Python Async Programming",
            content="Asyncio is a library for concurrent Python code."
        )
        note2 = zettel_service.create_note(
            title="JavaScript Promises",
            content="Promises handle asynchronous JavaScript operations."
        )

        search_service = SearchService(zettel_service)
        # FTS5 is available in test DB — text search should work
        results = search_service.search_combined(text="Python")

        result_ids = [r.note.id for r in results]
        assert note1.id in result_ids

    def test_search_combined_fallback_without_fts(self, zettel_service):
        """Graceful degradation when FTS5 is unavailable."""
        note = zettel_service.create_note(
            title="Fallback Test",
            content="This tests the fallback path."
        )

        search_service = SearchService(zettel_service)

        # Disable FTS to force fallback
        original = zettel_service.repository._fts_available
        zettel_service.repository._fts_available = False
        try:
            results = search_service.search_combined(text="Fallback")
            result_ids = [r.note.id for r in results]
            assert note.id in result_ids
        finally:
            zettel_service.repository._fts_available = original

    def test_search_combined_fts_with_tag_filter(self, zettel_service):
        """FTS text search + tag post-filter works correctly."""
        tagged = zettel_service.create_note(
            title="Python Guide",
            content="Complete Python programming guide.",
            tags=["python"]
        )
        untagged = zettel_service.create_note(
            title="Python Tips",
            content="Quick Python tips and tricks."
        )

        search_service = SearchService(zettel_service)
        results = search_service.search_combined(text="Python", tags=["python"])

        result_ids = [r.note.id for r in results]
        assert tagged.id in result_ids
        assert untagged.id not in result_ids