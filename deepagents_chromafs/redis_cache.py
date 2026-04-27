"""RedisContentCache: Redis-backed content cache for multi-session deployments.

Drop-in replacement for the default in-memory ``ContentCache``.  Plug it into
``ChromaFsBackend`` to share cached page content across multiple workers or
sessions, matching the original ChromaFs design which used Redis for this layer.

Usage::

    import redis
    from deepagents_chromafs import ChromaFsBackend
    from deepagents_chromafs.redis_cache import RedisContentCache

    client = redis.Redis(host="localhost", port=6379, db=0)
    cache = RedisContentCache(client, prefix="myapp", ttl=3600)

    backend = ChromaFsBackend(collection, cache=cache)

Requires the ``redis`` extra::

    pip install deepagents-chromafs[redis]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator

from deepagents_chromafs.cache import ContentCache


@runtime_checkable
class RedisClientProtocol(Protocol):
    """Structural interface covering the redis-py methods used by RedisContentCache.

    Any object that implements these methods (``redis.Redis``, ``fakeredis``,
    custom stubs) is accepted without requiring ``redis-py`` to be installed.
    """

    def get(self, key: str) -> bytes | None: ...  # noqa: D102
    def set(self, key: str, value: str) -> None: ...  # noqa: D102
    def setex(self, key: str, time: int, value: str) -> None: ...  # noqa: D102
    def exists(self, key: str) -> int: ...  # noqa: D102
    def delete(self, *keys: str) -> int: ...  # noqa: D102
    def scan_iter(self, match: str = "*") -> Iterator[str]: ...  # noqa: D102


class RedisContentCache(ContentCache):
    """Redis-backed content cache for shared, multi-session deployments.

    Each slug is stored as a Redis string key with an optional TTL.  Key names
    are namespaced by ``prefix`` to avoid collisions when the same Redis
    instance is shared across multiple collections or applications.

    Requires ``redis-py`` (``pip install deepagents-chromafs[redis]``).

    Args:
        client: A ``redis.Redis`` (or compatible) client instance.
        prefix: Namespace prefix for all keys.  Keys take the form
            ``{prefix}:{slug}``.
        ttl: Time-to-live in seconds for each cached entry.  ``0`` means
            no expiry.
    """

    def __init__(
        self,
        client: RedisClientProtocol,
        *,
        prefix: str = "chromafs",
        ttl: int = 3600,
    ) -> None:
        """Initialize the Redis cache.

        Args:
            client: ``redis.Redis`` (or compatible) client.
            prefix: Key namespace prefix.
            ttl: Entry TTL in seconds.  ``0`` disables expiry.
        """
        self._client = client
        self._prefix = prefix
        self._ttl = ttl

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key(self, slug: str) -> str:
        """Build the namespaced Redis key for a slug.

        Args:
            slug: Page slug (e.g. ``auth/oauth.md``).

        Returns:
            Redis key string.
        """
        return f"{self._prefix}:{slug}"

    # ------------------------------------------------------------------
    # ContentCache interface
    # ------------------------------------------------------------------

    def get(self, slug: str) -> str | None:
        """Return cached page content, or None if not present or expired.

        Args:
            slug: Page slug.

        Returns:
            Cached page text, or ``None`` on cache miss.
        """
        raw = self._client.get(self._key(slug))
        if raw is None:
            return None
        return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)

    def put(self, slug: str, content: str) -> None:
        """Store page content in Redis with the configured TTL.

        Args:
            slug: Page slug.
            content: Full reassembled page text.
        """
        key = self._key(slug)
        if self._ttl > 0:
            self._client.setex(key, self._ttl, content)
        else:
            self._client.set(key, content)

    def has(self, slug: str) -> bool:
        """Return True if the slug exists in Redis (and has not expired).

        Args:
            slug: Page slug.

        Returns:
            True when the key is present.
        """
        return bool(self._client.exists(self._key(slug)))

    def clear(self) -> None:
        """Delete all keys under this cache's prefix using SCAN.

        Uses ``SCAN`` (non-blocking) rather than ``KEYS`` to avoid stalling
        the Redis event loop on large keyspaces.
        """
        pattern = f"{self._prefix}:*"
        keys = list(self._client.scan_iter(match=pattern))
        if keys:
            self._client.delete(*keys)

    def __len__(self) -> int:
        """Return the number of cached entries under this prefix."""
        return sum(1 for _ in self._client.scan_iter(match=f"{self._prefix}:*"))
