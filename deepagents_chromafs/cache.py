"""ContentCache: in-memory per-session cache for reassembled page content.

Prevents redundant Chroma fetches when the same page is accessed multiple
times in a single session (e.g. grep prefetch followed by a cat).
"""

from __future__ import annotations


class ContentCache:
    """Thread-unsafe in-memory store mapping page slugs to full page text.

    Designed for single-session use.  Storing content here means a ``cat``
    following a ``grep`` never hits the database a second time for the same
    page.

    For multi-worker deployments a shared cache (e.g. Redis) can be wired in
    by subclassing and overriding ``get`` / ``put``.
    """

    def __init__(self) -> None:
        """Initialize an empty cache."""
        self._store: dict[str, str] = {}

    def get(self, slug: str) -> str | None:
        """Return the cached page content for a slug, or None if not cached.

        Args:
            slug: Chroma page slug (e.g. ``auth/oauth.md``).

        Returns:
            Full reassembled page text, or ``None`` if the slug is not cached.
        """
        return self._store.get(slug)

    def put(self, slug: str, content: str) -> None:
        """Store reassembled page content.

        Args:
            slug: Chroma page slug.
            content: Full page text (all chunks joined in order).
        """
        self._store[slug] = content

    def has(self, slug: str) -> bool:
        """Return True if the slug has a cached entry.

        Args:
            slug: Chroma page slug.

        Returns:
            True when the slug is already in the cache.
        """
        return slug in self._store

    def clear(self) -> None:
        """Remove all cached entries."""
        self._store.clear()

    def __len__(self) -> int:
        """Return the number of cached entries."""
        return len(self._store)
