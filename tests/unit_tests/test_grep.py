"""Unit tests for the grep pipeline (grep.py)."""

from __future__ import annotations

from deepagents_chromafs.cache import ContentCache
from deepagents_chromafs.grep import (
    bulk_prefetch,
    fetch_page,
    find_matching_slugs,
    fine_filter,
    parse_groups_metadata,
    to_chroma_filter,
)
from tests.unit_tests.fake_collection import FakeCollection

# ---------------------------------------------------------------------------
# to_chroma_filter
# ---------------------------------------------------------------------------


def test_to_chroma_filter_fixed_string():
    f = to_chroma_filter("hello world", fixed_string=True)
    assert f == {"$contains": "hello world"}


def test_to_chroma_filter_fixed_string_ignore_case():
    f = to_chroma_filter("hello", fixed_string=True, ignore_case=True)
    # Must be a $regex with (?i) prefix and escaped pattern
    assert f["$regex"].startswith("(?i)")
    assert "hello" in f["$regex"]


def test_to_chroma_filter_regex():
    f = to_chroma_filter(r"\bfoo\b", fixed_string=False)
    assert f == {"$regex": r"\bfoo\b"}


def test_to_chroma_filter_regex_ignore_case():
    f = to_chroma_filter(r"\bfoo\b", fixed_string=False, ignore_case=True)
    assert f["$regex"].startswith("(?i)")


def test_to_chroma_filter_special_chars_escaped():
    f = to_chroma_filter("foo.bar[0]", fixed_string=True)
    # Should use $contains with literal string
    assert f == {"$contains": "foo.bar[0]"}


# ---------------------------------------------------------------------------
# find_matching_slugs
# ---------------------------------------------------------------------------


def _make_chunk_doc(slug: str, chunk_index: int, text: str) -> dict:
    return {
        "id": f"{slug}:{chunk_index}",
        "document": text,
        "page_slug": slug,
        "chunk_index": chunk_index,
    }


def test_find_matching_slugs_basic():
    collection = FakeCollection([
        _make_chunk_doc("auth/oauth", 0, "This is the OAuth2 guide."),
        _make_chunk_doc("auth/api-keys", 0, "API keys are used for authentication."),
        _make_chunk_doc("billing/plans", 0, "We offer multiple billing plans."),
    ])
    chroma_filter = to_chroma_filter("OAuth2")
    matched = find_matching_slugs(collection, chroma_filter, ["auth/oauth", "auth/api-keys", "billing/plans"])
    assert matched == ["auth/oauth"]


def test_find_matching_slugs_empty_candidates():
    collection = FakeCollection([
        _make_chunk_doc("auth/oauth", 0, "OAuth2 guide here"),
    ])
    chroma_filter = to_chroma_filter("OAuth2")
    matched = find_matching_slugs(collection, chroma_filter, [])
    # Empty candidates → search full collection
    assert "auth/oauth" in matched


def test_find_matching_slugs_no_match():
    collection = FakeCollection([
        _make_chunk_doc("auth/oauth", 0, "Nothing relevant here"),
    ])
    chroma_filter = to_chroma_filter("XYZZY_NOT_PRESENT")
    matched = find_matching_slugs(collection, chroma_filter, ["auth/oauth"])
    assert matched == []


def test_find_matching_slugs_deduplicates():
    # Same slug appears in two chunks that both match
    collection = FakeCollection([
        _make_chunk_doc("auth/oauth", 0, "OAuth2 flow explained"),
        _make_chunk_doc("auth/oauth", 1, "OAuth2 refresh tokens"),
    ])
    chroma_filter = to_chroma_filter("OAuth2")
    matched = find_matching_slugs(collection, chroma_filter, ["auth/oauth"])
    assert matched == ["auth/oauth"]  # de-duplicated


# ---------------------------------------------------------------------------
# fetch_page
# ---------------------------------------------------------------------------


def test_fetch_page_single_chunk():
    collection = FakeCollection([
        _make_chunk_doc("auth/oauth", 0, "Full OAuth guide content"),
    ])
    content = fetch_page(collection, "auth/oauth")
    assert content == "Full OAuth guide content"


def test_fetch_page_multiple_chunks_ordered():
    collection = FakeCollection([
        _make_chunk_doc("auth/oauth", 1, "Second chunk"),
        _make_chunk_doc("auth/oauth", 0, "First chunk"),
    ])
    content = fetch_page(collection, "auth/oauth")
    assert content.startswith("First chunk")
    assert "Second chunk" in content


def test_fetch_page_missing_slug():
    collection = FakeCollection([])
    content = fetch_page(collection, "missing/slug")
    assert content == ""


# ---------------------------------------------------------------------------
# bulk_prefetch
# ---------------------------------------------------------------------------


def test_bulk_prefetch_populates_cache():
    collection = FakeCollection([
        _make_chunk_doc("auth/oauth", 0, "OAuth content"),
        _make_chunk_doc("billing/plans", 0, "Billing content"),
    ])
    cache = ContentCache()
    bulk_prefetch(collection, ["auth/oauth", "billing/plans"], cache)
    assert cache.has("auth/oauth")
    assert cache.has("billing/plans")
    assert cache.get("auth/oauth") == "OAuth content"


def test_bulk_prefetch_skips_cached():
    collection = FakeCollection([
        _make_chunk_doc("auth/oauth", 0, "OAuth content"),
    ])
    cache = ContentCache()
    cache.put("auth/oauth", "ALREADY CACHED")

    bulk_prefetch(collection, ["auth/oauth"], cache)
    # Cache should not be overwritten
    assert cache.get("auth/oauth") == "ALREADY CACHED"


# ---------------------------------------------------------------------------
# fine_filter
# ---------------------------------------------------------------------------


def test_fine_filter_returns_matches():
    cache = ContentCache()
    cache.put("auth/oauth", "line one\nOAuth2 login flow\nline three")

    matches = fine_filter("OAuth2", ["auth/oauth"], cache, fixed_string=True)
    assert len(matches) == 1
    assert matches[0]["path"] == "/auth/oauth"
    assert matches[0]["line"] == 2
    assert "OAuth2 login flow" in matches[0]["text"]


def test_fine_filter_no_match():
    cache = ContentCache()
    cache.put("auth/oauth", "nothing here")
    matches = fine_filter("XYZZY", ["auth/oauth"], cache, fixed_string=True)
    assert matches == []


def test_fine_filter_case_insensitive():
    cache = ContentCache()
    cache.put("auth/oauth", "OAUTH2 FLOW\noauth2 flow\nOAuth2 Flow")
    matches = fine_filter("oauth2", ["auth/oauth"], cache, fixed_string=True, ignore_case=True)
    assert len(matches) == 3


def test_fine_filter_case_sensitive_by_default():
    cache = ContentCache()
    cache.put("auth/oauth", "OAUTH2 FLOW\noauth2 flow")
    matches = fine_filter("oauth2", ["auth/oauth"], cache, fixed_string=True, ignore_case=False)
    # Only lowercase line matches
    assert len(matches) == 1
    assert matches[0]["line"] == 2


def test_fine_filter_missing_from_cache():
    cache = ContentCache()
    # Slug not in cache — should be silently skipped
    matches = fine_filter("anything", ["missing/slug"], cache)
    assert matches == []


def test_fine_filter_custom_slug_to_path():
    cache = ContentCache()
    cache.put("docs/intro", "Introduction text here")
    matches = fine_filter(
        "Introduction",
        ["docs/intro"],
        cache,
        slug_to_path=lambda s: "/custom/" + s,
    )
    assert matches[0]["path"] == "/custom/docs/intro"


def test_fine_filter_multiple_slugs():
    cache = ContentCache()
    cache.put("auth/oauth", "OAuth2 is here")
    cache.put("billing/plans", "OAuth2 pricing plans")
    cache.put("public/home", "Welcome home")

    matches = fine_filter("OAuth2", ["auth/oauth", "billing/plans", "public/home"], cache)
    paths = {m["path"] for m in matches}
    assert "/auth/oauth" in paths
    assert "/billing/plans" in paths
    assert "/public/home" not in paths


# ---------------------------------------------------------------------------
# parse_groups_metadata
# ---------------------------------------------------------------------------


def test_parse_groups_json_string():
    assert parse_groups_metadata('["admin", "billing"]') == ["admin", "billing"]


def test_parse_groups_list():
    assert parse_groups_metadata(["admin"]) == ["admin"]


def test_parse_groups_none():
    assert parse_groups_metadata(None) == []


def test_parse_groups_invalid_json():
    assert parse_groups_metadata("not json") == []
