"""Grep pipeline for ChromaFs.

Implements the 4-step grep optimisation from the ChromaFs algorithm:

1. **to_chroma_filter** — convert a search pattern + flags into a Chroma
   ``where_document`` filter dict (``$contains`` for fixed strings,
   ``$regex`` for patterns, with case-insensitive variants).

2. **find_matching_slugs** — coarse filter: ask Chroma which chunks contain
   the pattern, scoped to a candidate slug list derived from the in-memory
   tree.  Returns the de-duplicated set of matching page slugs.

3. **bulk_prefetch** — pull all chunks for each matched slug into the
   ``ContentCache`` concurrently (sync: sequential; async: gathered).

4. **fine_filter** — run an in-memory Python ``re`` search on the cached
   page content to produce structured ``GrepMatch`` results with exact line
   numbers, matching the ``BackendProtocol`` contract.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import chromadb
    from deepagents.backends.protocol import GrepMatch

    from deepagents_chromafs.cache import ContentCache

# Maximum number of slugs per Chroma $in batch to avoid overly large queries.
_SLUG_BATCH_SIZE = 500


# ---------------------------------------------------------------------------
# Step 1 — build the Chroma where_document filter
# ---------------------------------------------------------------------------


def to_chroma_filter(
    pattern: str,
    *,
    fixed_string: bool = True,
    ignore_case: bool = False,
) -> dict[str, object]:
    """Convert a search pattern to a Chroma ``where_document`` filter.

    Chroma supports two document-content operators:

    - ``$contains``: case-sensitive literal substring match.
    - ``$regex``: RE2-compatible regular expression match.

    When ``ignore_case=True`` and ``fixed_string=True``, the pattern is
    promoted to a case-insensitive regex (``(?i)re.escape(pattern)``).

    Args:
        pattern: Search string or regex pattern.
        fixed_string: When ``True`` (default), treat ``pattern`` as a literal
            string rather than a regular expression.
        ignore_case: When ``True``, perform case-insensitive matching.

    Returns:
        A Chroma ``where_document`` filter dict.
    """
    if fixed_string and not ignore_case:
        # Fast path: literal case-sensitive substring — use $contains
        return {"$contains": pattern}

    if fixed_string and ignore_case:
        # Promote to regex so we can add (?i) flag
        return {"$regex": f"(?i){re.escape(pattern)}"}

    if ignore_case:
        # Regex pattern with case-insensitive flag
        return {"$regex": f"(?i){pattern}"}

    return {"$regex": pattern}


# ---------------------------------------------------------------------------
# Step 2 — coarse filter: find slugs whose chunks match the pattern
# ---------------------------------------------------------------------------


def _query_chroma_batch(
    collection: chromadb.Collection,
    chroma_filter: dict[str, object],
    slug_batch: list[str],
    slug_field: str,
) -> list[str]:
    """Run a single Chroma get() call for one batch of candidate slugs.

    Args:
        collection: ChromaDB collection to query.
        chroma_filter: ``where_document`` filter dict from ``to_chroma_filter``.
        slug_batch: Subset of candidate slugs to scope this query.
        slug_field: Metadata field name that stores the page slug.

    Returns:
        List of slugs (possibly with duplicates across chunks) that matched.
    """
    where: dict[str, object] = {slug_field: {"$in": slug_batch}}
    results = collection.get(
        where=where,
        where_document=chroma_filter,
        include=["metadatas"],
    )
    metadatas: list[dict[str, object]] = results.get("metadatas") or []
    return [str(m[slug_field]) for m in metadatas if slug_field in m]


def find_matching_slugs(
    collection: chromadb.Collection,
    chroma_filter: dict[str, object],
    candidate_slugs: list[str],
    *,
    slug_field: str = "page_slug",
) -> list[str]:
    """Coarse filter: return slugs of pages that contain the pattern.

    Queries Chroma with the given ``where_document`` filter, scoped to
    ``candidate_slugs``.  Large slug lists are batched to avoid oversized
    queries.

    Args:
        collection: ChromaDB collection to query.
        chroma_filter: ``where_document`` filter from ``to_chroma_filter``.
        candidate_slugs: Slugs to search within (from the in-memory tree).
            Pass an empty list to search the entire collection.
        slug_field: Metadata field name for the page slug.

    Returns:
        De-duplicated, sorted list of matching page slugs.
    """
    if not candidate_slugs:
        # No candidates scoped — search entire collection
        results = collection.get(
            where_document=chroma_filter,
            include=["metadatas"],
        )
        metadatas: list[dict[str, object]] = results.get("metadatas") or []
        return sorted({str(m[slug_field]) for m in metadatas if slug_field in m})

    matched: set[str] = set()
    for i in range(0, len(candidate_slugs), _SLUG_BATCH_SIZE):
        batch = candidate_slugs[i : i + _SLUG_BATCH_SIZE]
        matched.update(_query_chroma_batch(collection, chroma_filter, batch, slug_field))

    return sorted(matched)


# ---------------------------------------------------------------------------
# Step 3 — bulk prefetch: reassemble pages into the cache
# ---------------------------------------------------------------------------


def fetch_page(
    collection: chromadb.Collection,
    slug: str,
    *,
    slug_field: str = "page_slug",
    chunk_index_field: str = "chunk_index",
) -> str:
    """Fetch all chunks for a page slug and return the reassembled text.

    Retrieves every chunk whose ``slug_field`` metadata equals ``slug``,
    sorts them by ``chunk_index_field``, and joins the documents with
    newlines.

    Args:
        collection: ChromaDB collection.
        slug: Page slug to fetch (e.g. ``auth/oauth.md``).
        slug_field: Metadata field name for the page slug.
        chunk_index_field: Metadata field name for the chunk index.

    Returns:
        Full page text with chunks joined in order.
    """
    results = collection.get(
        where={slug_field: slug},
        include=["documents", "metadatas"],
    )
    documents: list[str] = results.get("documents") or []
    metadatas: list[dict[str, object]] = results.get("metadatas") or []

    if not documents:
        return ""

    # Pair each chunk with its index and sort
    indexed: list[tuple[int, str]] = []
    for doc, meta in zip(documents, metadatas, strict=False):
        idx = int(meta.get(chunk_index_field, 0))
        indexed.append((idx, doc))

    indexed.sort(key=lambda x: x[0])
    return "\n".join(text for _, text in indexed)


def bulk_prefetch(
    collection: chromadb.Collection,
    slugs: list[str],
    cache: ContentCache,
    *,
    slug_field: str = "page_slug",
    chunk_index_field: str = "chunk_index",
) -> None:
    """Prefetch and cache page content for each slug (synchronous).

    Slugs already present in ``cache`` are skipped.

    Args:
        collection: ChromaDB collection.
        slugs: Page slugs to prefetch.
        cache: Target content cache.
        slug_field: Metadata field name for the page slug.
        chunk_index_field: Metadata field name for the chunk index.
    """
    for slug in slugs:
        if cache.has(slug):
            continue
        content = fetch_page(
            collection,
            slug,
            slug_field=slug_field,
            chunk_index_field=chunk_index_field,
        )
        cache.put(slug, content)


async def abulk_prefetch(
    collection: chromadb.Collection,
    slugs: list[str],
    cache: ContentCache,
    *,
    slug_field: str = "page_slug",
    chunk_index_field: str = "chunk_index",
) -> None:
    """Async version of ``bulk_prefetch`` — fetches slugs concurrently.

    Each slug fetch runs in a thread pool via ``asyncio.to_thread`` so the
    event loop is not blocked.  Slugs already in ``cache`` are skipped.

    Args:
        collection: ChromaDB collection.
        slugs: Page slugs to prefetch.
        cache: Target content cache.
        slug_field: Metadata field name for the page slug.
        chunk_index_field: Metadata field name for the chunk index.
    """
    missing = [s for s in slugs if not cache.has(s)]
    if not missing:
        return

    async def _fetch_one(slug: str) -> tuple[str, str]:
        content = await asyncio.to_thread(
            fetch_page,
            collection,
            slug,
            slug_field=slug_field,
            chunk_index_field=chunk_index_field,
        )
        return slug, content

    results = await asyncio.gather(*[_fetch_one(s) for s in missing])
    for slug, content in results:
        cache.put(slug, content)


# ---------------------------------------------------------------------------
# Step 4 — fine filter: in-memory regex on cached content
# ---------------------------------------------------------------------------


def fine_filter(
    pattern: str,
    slugs: list[str],
    cache: ContentCache,
    *,
    fixed_string: bool = True,
    ignore_case: bool = False,
    slug_to_path: object = None,
) -> list[GrepMatch]:
    """Run in-memory regex search on cached page content.

    Produces line-level ``GrepMatch`` results with 1-indexed line numbers.

    Args:
        pattern: Search string or regex pattern.
        slugs: Page slugs to search (must already be in ``cache``).
        cache: Content cache populated by ``bulk_prefetch``.
        fixed_string: When ``True``, escape ``pattern`` as a literal string.
        ignore_case: When ``True``, perform case-insensitive matching.
        slug_to_path: Optional callable ``(slug: str) -> str`` that converts
            a slug to a virtual filesystem path for the ``GrepMatch.path``
            field.  Defaults to ``"/" + slug``.

    Returns:
        List of ``GrepMatch`` dicts with ``path``, ``line``, and ``text``.
    """
    flags = re.IGNORECASE if ignore_case else 0
    effective_pattern = re.escape(pattern) if fixed_string else pattern

    try:
        regex = re.compile(effective_pattern, flags)
    except re.error:
        return []

    def default_slug_to_path(slug: str) -> str:
        return "/" + slug

    resolve_path = slug_to_path if callable(slug_to_path) else default_slug_to_path

    matches: list[GrepMatch] = []
    for slug in slugs:
        content = cache.get(slug)
        if content is None:
            continue
        path = resolve_path(slug)
        for line_num, line in enumerate(content.splitlines(), start=1):
            if regex.search(line):
                matches.append({"path": path, "line": line_num, "text": line})

    return matches


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def parse_groups_metadata(raw: str | list[str] | None) -> list[str]:
    """Parse the ``groups`` metadata field stored in Chroma.

    Chroma does not support list metadata values, so groups are stored as
    a JSON-serialised string (e.g. ``'["admin", "billing"]'``).  This
    helper handles both serialised strings and bare list values for
    forwards-compatibility.

    Args:
        raw: Raw metadata value for the ``groups`` field.

    Returns:
        List of group names.  Returns an empty list on any parse failure.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(g) for g in raw]
    try:
        parsed = json.loads(raw)
        return [str(g) for g in parsed] if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []
