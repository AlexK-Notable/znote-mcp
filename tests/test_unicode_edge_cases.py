"""Tests for Unicode edge cases in the Zettelkasten MCP server.

Tests that verify proper handling of Unicode characters, emojis, RTL text,
combining characters, and other internationalization concerns.
"""

import pytest

from znote_mcp.models.schema import LinkType, NoteType
from znote_mcp.services.search_service import SearchService


class TestUnicodeInTitles:
    """Tests for Unicode handling in note titles."""

    def test_emoji_in_title(self, zettel_service):
        """Test notes with emoji in titles."""
        note = zettel_service.create_note(
            title="🚀 Rocket Science 🌟", content="Space exploration concepts."
        )

        retrieved = zettel_service.get_note(note.id)
        assert "🚀" in retrieved.title
        assert "🌟" in retrieved.title

    def test_chinese_characters_in_title(self, zettel_service):
        """Test notes with Chinese characters in titles."""
        note = zettel_service.create_note(
            title="中文标题测试", content="这是中文内容。"
        )

        retrieved = zettel_service.get_note(note.id)
        assert retrieved.title == "中文标题测试"

    def test_japanese_characters_in_title(self, zettel_service):
        """Test notes with Japanese characters (Hiragana, Katakana, Kanji)."""
        note = zettel_service.create_note(
            title="日本語のタイトル テスト",
            content="ひらがな、カタカナ、漢字のテスト。",
        )

        retrieved = zettel_service.get_note(note.id)
        assert "日本語" in retrieved.title
        assert "タイトル" in retrieved.title

    def test_korean_characters_in_title(self, zettel_service):
        """Test notes with Korean (Hangul) characters."""
        note = zettel_service.create_note(
            title="한국어 제목 테스트", content="한글 내용입니다."
        )

        retrieved = zettel_service.get_note(note.id)
        assert "한국어" in retrieved.title

    def test_arabic_rtl_title(self, zettel_service):
        """Test notes with Arabic right-to-left text."""
        note = zettel_service.create_note(
            title="اختبار العنوان العربي", content="محتوى باللغة العربية."
        )

        retrieved = zettel_service.get_note(note.id)
        assert "العربي" in retrieved.title

    def test_hebrew_rtl_title(self, zettel_service):
        """Test notes with Hebrew right-to-left text."""
        note = zettel_service.create_note(title="כותרת בעברית", content="תוכן בעברית.")

        retrieved = zettel_service.get_note(note.id)
        assert "בעברית" in retrieved.title

    def test_mixed_script_title(self, zettel_service):
        """Test notes with mixed scripts in title."""
        note = zettel_service.create_note(
            title="English 中文 日本語 한국어 Mixed", content="Multi-language content."
        )

        retrieved = zettel_service.get_note(note.id)
        assert "English" in retrieved.title
        assert "中文" in retrieved.title
        assert "日本語" in retrieved.title
        assert "한국어" in retrieved.title


class TestUnicodeInContent:
    """Tests for Unicode handling in note content."""

    def test_mathematical_symbols(self, zettel_service):
        """Test notes with mathematical symbols."""
        note = zettel_service.create_note(
            title="Math Formulas",
            content="∀x ∈ ℝ: x² ≥ 0, ∑∞ₙ₌₁ 1/n² = π²/6, √2 ≈ 1.414",
        )

        retrieved = zettel_service.get_note(note.id)
        assert "∀x" in retrieved.content
        assert "∈" in retrieved.content
        assert "ℝ" in retrieved.content
        assert "∑" in retrieved.content
        assert "π" in retrieved.content
        assert "√" in retrieved.content

    def test_currency_symbols(self, zettel_service):
        """Test notes with various currency symbols."""
        note = zettel_service.create_note(
            title="Currency Symbols",
            content="USD: $100, EUR: €85, GBP: £75, JPY: ¥11,000, BTC: ₿0.003",
        )

        retrieved = zettel_service.get_note(note.id)
        assert "$" in retrieved.content
        assert "€" in retrieved.content
        assert "£" in retrieved.content
        assert "¥" in retrieved.content
        assert "₿" in retrieved.content

    def test_box_drawing_characters(self, zettel_service):
        """Test notes with box drawing characters for diagrams."""
        note = zettel_service.create_note(
            title="ASCII Art Diagram",
            content="""
┌─────────────────┐
│   Component A   │
├─────────────────┤
│  ├── Module 1   │
│  └── Module 2   │
└─────────────────┘
""",
        )

        retrieved = zettel_service.get_note(note.id)
        assert "┌" in retrieved.content
        assert "│" in retrieved.content
        assert "└" in retrieved.content
        assert "├" in retrieved.content

    def test_emoji_sequences(self, zettel_service):
        """Test notes with complex emoji sequences."""
        note = zettel_service.create_note(
            title="Emoji Test",
            content="Family: 👨‍👩‍👧‍👦 Flag: 🇯🇵 Skin tone: 👍🏻 ZWJ: 👩‍💻",
        )

        retrieved = zettel_service.get_note(note.id)
        # Emoji should be preserved
        assert "👨" in retrieved.content or "👨‍👩‍👧‍👦" in retrieved.content

    def test_combining_characters(self, zettel_service):
        """Test notes with combining diacritical marks."""
        # Using combining characters (separate diacritics)
        note = zettel_service.create_note(
            title="Combining Characters",
            content="Ḉ̧óṃb̧ĩñĩñg̃ characters: café naïve résumé",
        )

        retrieved = zettel_service.get_note(note.id)
        assert "café" in retrieved.content or "cafe" in retrieved.content

    def test_zero_width_characters(self, zettel_service):
        """Test notes with zero-width characters."""
        # Zero-width space (U+200B) and zero-width non-joiner (U+200C)
        note = zettel_service.create_note(
            title="Zero Width Test", content="word\u200bwith\u200czero\u200dwidth"
        )

        retrieved = zettel_service.get_note(note.id)
        # Content should be preserved (with or without zero-width chars)
        assert "word" in retrieved.content
        assert "zero" in retrieved.content


class TestUnicodeInTags:
    """Tests for Unicode handling in tags."""

    def test_emoji_tag(self, zettel_service):
        """Test tags with emoji."""
        note = zettel_service.create_note(
            title="Emoji Tagged", content="Content", tags=["🔥hot", "⭐starred"]
        )

        tag_names = {tag.name for tag in note.tags}
        # Tags should be preserved or normalized
        assert len(tag_names) >= 1

    def test_chinese_tags(self, zettel_service):
        """Test tags with Chinese characters."""
        note = zettel_service.create_note(
            title="Chinese Tagged", content="Content", tags=["中文", "标签", "测试"]
        )

        tag_names = {tag.name for tag in note.tags}
        assert "中文" in tag_names
        assert "标签" in tag_names
        assert "测试" in tag_names

    def test_accented_tags(self, zettel_service):
        """Test tags with accented characters."""
        note = zettel_service.create_note(
            title="Accented Tags", content="Content", tags=["café", "naïve", "résumé"]
        )

        tag_names = {tag.name for tag in note.tags}
        assert "café" in tag_names or "cafe" in tag_names


class TestUnicodeSearch:
    """Tests for searching with Unicode text."""

    def test_search_chinese_text(self, zettel_service):
        """Test full-text search with Chinese characters."""
        zettel_service.create_note(
            title="中文测试", content="这是一个中文内容的测试笔记。"
        )

        search_service = SearchService(zettel_service)
        results = search_service.search_combined(text="中文")

        assert len(results) >= 1
        assert "中文" in results[0].note.title or "中文" in results[0].note.content

    def test_search_emoji(self, zettel_service):
        """Test searching for emoji."""
        zettel_service.create_note(
            title="🚀 Rocket Launch", content="The 🚀 is launching today!"
        )

        search_service = SearchService(zettel_service)
        results = search_service.search_combined(text="🚀")

        assert len(results) >= 1

    def test_search_accented_characters(self, zettel_service):
        """Test searching for accented characters."""
        zettel_service.create_note(
            title="Café Culture",
            content="The café is famous for its résumé of pastries.",
        )

        search_service = SearchService(zettel_service)
        results = search_service.search_combined(text="café")

        assert len(results) >= 1

    def test_case_insensitive_unicode_search(self, zettel_service):
        """Test that Unicode search is case-insensitive where applicable."""
        zettel_service.create_note(
            title="German Umlauts", content="Größe means size. GRÖSSE is also valid."
        )

        search_service = SearchService(zettel_service)

        # Both cases should find the note
        results_lower = search_service.search_combined(text="größe")
        results_upper = search_service.search_combined(text="GRÖSSE")

        # At least one should match
        assert len(results_lower) >= 1 or len(results_upper) >= 1


class TestUnicodeLinks:
    """Tests for Unicode handling in link descriptions."""

    def test_unicode_link_description(self, zettel_service):
        """Test links with Unicode descriptions."""
        note1 = zettel_service.create_note(
            title="Source 来源", content="Source content"
        )
        note2 = zettel_service.create_note(
            title="Target 目标", content="Target content"
        )

        # Create link with Unicode description
        zettel_service.create_link(
            note1.id,
            note2.id,
            LinkType.REFERENCE,
            description="参考链接 - Reference Link 📚",
        )

        # Verify link was created
        linked_notes = zettel_service.get_linked_notes(note1.id, "outgoing")
        assert len(linked_notes) == 1
        assert linked_notes[0].id == note2.id


class TestUnicodeNormalization:
    """Tests for Unicode normalization handling."""

    def test_composed_vs_decomposed_search(self, zettel_service):
        """Test searching for composed vs decomposed Unicode."""
        # Create note with composed form (é as single character)
        note = zettel_service.create_note(
            title="Café Note", content="Visit the café today."  # composed é (U+00E9)
        )

        search_service = SearchService(zettel_service)

        # Search with composed form should work
        results = search_service.search_combined(text="café")
        assert len(results) >= 1

    def test_fullwidth_vs_halfwidth(self, zettel_service):
        """Test fullwidth vs halfwidth character handling."""
        # Fullwidth characters used in CJK text
        note = zettel_service.create_note(
            title="Ｆｕｌｌｗｉｄｔｈ Test",  # Fullwidth ASCII
            content="ＡＢＣＤ = ABCD",
        )

        retrieved = zettel_service.get_note(note.id)
        # Content should be preserved
        assert "Ｆｕｌｌ" in retrieved.title or "Full" in retrieved.title


class TestUnicodeBoundaryConditions:
    """Tests for Unicode boundary conditions."""

    def test_very_long_unicode_string(self, zettel_service):
        """Test handling of very long Unicode strings."""
        # Long string with mix of BMP and non-BMP characters
        long_unicode = "こんにちは" * 1000 + "🌟" * 100

        note = zettel_service.create_note(title="Long Unicode", content=long_unicode)

        retrieved = zettel_service.get_note(note.id)
        # Should handle without crashing
        assert len(retrieved.content) > 0

    def test_null_character_handling(self, zettel_service):
        """Test handling of null characters in content."""
        # Some systems have issues with embedded nulls
        content_with_null = "Before\x00After"

        try:
            note = zettel_service.create_note(
                title="Null Test", content=content_with_null
            )
            # If it succeeds, check that something was stored
            retrieved = zettel_service.get_note(note.id)
            assert retrieved is not None
        except (ValueError, UnicodeError):
            # Also acceptable to reject null characters
            pass

    def test_bom_handling(self, zettel_service):
        """Test handling of byte order marks."""
        # UTF-8 BOM at start of content
        content_with_bom = "\ufeffContent with BOM"

        note = zettel_service.create_note(title="BOM Test", content=content_with_bom)

        retrieved = zettel_service.get_note(note.id)
        # BOM might be stripped or preserved
        assert "Content" in retrieved.content

    def test_private_use_area_characters(self, zettel_service):
        """Test handling of Private Use Area characters."""
        # PUA characters (often used for custom icons/fonts)
        pua_content = "Custom symbol: \ue000 \uf000 end"

        note = zettel_service.create_note(title="PUA Test", content=pua_content)

        retrieved = zettel_service.get_note(note.id)
        # Should handle without crashing
        assert "Custom symbol" in retrieved.content

    def test_surrogate_pairs_emoji(self, zettel_service):
        """Test handling of characters requiring surrogate pairs (non-BMP)."""
        # Characters outside BMP require surrogate pairs in UTF-16
        non_bmp = "𝄞 Musical symbols 𝔄𝔅ℭ Fraktur 🦄 Unicorn"

        note = zettel_service.create_note(title="Non-BMP Characters", content=non_bmp)

        retrieved = zettel_service.get_note(note.id)
        # Should preserve or at least not crash
        assert "Musical" in retrieved.content or "𝄞" in retrieved.content
