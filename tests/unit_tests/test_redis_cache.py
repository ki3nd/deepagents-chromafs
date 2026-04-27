"""Unit tests for RedisContentCache using a fake Redis client."""

from __future__ import annotations

import json

from deepagents_chromafs.backend import ChromaFsBackend
from deepagents_chromafs.cache import ContentCache
from deepagents_chromafs.redis_cache import RedisContentCache
from tests.unit_tests.fake_collection import FakeCollection

# ---------------------------------------------------------------------------
# Fake Redis client (no network, no dependency on redis-py)
# ---------------------------------------------------------------------------


class FakeRedis:
    """Minimal in-memory Redis stub covering the methods used by RedisContentCache."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, key: str) -> bytes | None:
        val = self._store.get(key)
        return val.encode() if val is not None else None

    def set(self, key: str, value: str) -> None:
        self._store[key] = value

    def setex(self, key: str, time: int, value: str) -> None:
        self._store[key] = value

    def exists(self, key: str) -> int:
        return 1 if key in self._store else 0

    def delete(self, *keys: str) -> int:
        count = 0
        for key in keys:
            if key in self._store:
                del self._store[key]
                count += 1
        return count

    def scan_iter(self, match: str = "*") -> list[str]:
        prefix = match.rstrip("*")
        return [k for k in self._store if k.startswith(prefix)]


# ---------------------------------------------------------------------------
# RedisContentCache — unit tests
# ---------------------------------------------------------------------------


def _cache(prefix: str = "test", ttl: int = 3600) -> tuple[RedisContentCache, FakeRedis]:
    client = FakeRedis()
    return RedisContentCache(client, prefix=prefix, ttl=ttl), client


def test_put_and_get():
    cache, _ = _cache()
    cache.put("auth/oauth.md", "OAuth2 content")
    assert cache.get("auth/oauth.md") == "OAuth2 content"


def test_get_miss_returns_none():
    cache, _ = _cache()
    assert cache.get("missing/slug.md") is None


def test_has_present():
    cache, _ = _cache()
    cache.put("docs/intro.md", "hello")
    assert cache.has("docs/intro.md")


def test_has_missing():
    cache, _ = _cache()
    assert not cache.has("docs/intro.md")


def test_key_uses_prefix():
    cache, client = _cache(prefix="myapp")
    cache.put("auth/oauth.md", "content")
    assert "myapp:auth/oauth.md" in client._store


def test_prefix_isolation():
    client = FakeRedis()
    cache_a = RedisContentCache(client, prefix="app_a")
    cache_b = RedisContentCache(client, prefix="app_b")
    cache_a.put("docs/intro.md", "from A")
    cache_b.put("docs/intro.md", "from B")
    assert cache_a.get("docs/intro.md") == "from A"
    assert cache_b.get("docs/intro.md") == "from B"


def test_clear_removes_only_own_prefix():
    client = FakeRedis()
    cache_a = RedisContentCache(client, prefix="app_a")
    cache_b = RedisContentCache(client, prefix="app_b")
    cache_a.put("docs/intro.md", "A")
    cache_b.put("docs/intro.md", "B")
    cache_a.clear()
    assert cache_a.get("docs/intro.md") is None
    assert cache_b.get("docs/intro.md") == "B"


def test_clear_empty_is_noop():
    cache, _ = _cache()
    cache.clear()  # should not raise


def test_len():
    cache, _ = _cache()
    assert len(cache) == 0
    cache.put("a.md", "x")
    cache.put("b.md", "y")
    assert len(cache) == 2
    cache.clear()
    assert len(cache) == 0


def test_put_no_ttl():
    """ttl=0 uses SET instead of SETEX."""
    cache, client = _cache(ttl=0)
    cache.put("docs/intro.md", "content")
    assert client._store.get("test:docs/intro.md") == "content"


def test_bytes_decoded():
    """get() always returns str even when Redis returns bytes."""
    cache, client = _cache()
    client._store["test:auth/oauth.md"] = "raw content"
    result = cache.get("auth/oauth.md")
    assert isinstance(result, str)
    assert result == "raw content"


# ---------------------------------------------------------------------------
# Integration: ChromaFsBackend with RedisContentCache
# ---------------------------------------------------------------------------

_TREE_DATA = {
    "auth/oauth.md": {"isPublic": True, "groups": []},
    "docs/intro.md": {"isPublic": True, "groups": []},
}


def _make_chunk(slug: str, idx: int, text: str) -> dict:
    return {"id": f"{slug}:{idx}", "document": text, "page_slug": slug, "chunk_index": idx}


def test_backend_uses_injected_redis_cache():
    """Backend reads from Redis cache on second access, not from Chroma."""
    col = FakeCollection([
        {"id": "__path_tree__", "document": json.dumps(_TREE_DATA)},
        _make_chunk("auth/oauth.md", 0, "OAuth2 guide"),
        _make_chunk("docs/intro.md", 0, "Welcome"),
    ])
    client = FakeRedis()
    cache = RedisContentCache(client, prefix="test")
    backend = ChromaFsBackend(col, cache=cache)

    result = backend.read("/auth/oauth.md")
    assert result.file_data["content"] == "OAuth2 guide"
    # Cache should now hold the slug
    assert cache.has("auth/oauth.md")


def test_backend_default_cache_is_in_memory():
    """Without cache= kwarg, backend creates its own in-memory ContentCache."""
    col = FakeCollection([
        {"id": "__path_tree__", "document": json.dumps(_TREE_DATA)},
        _make_chunk("auth/oauth.md", 0, "OAuth2 guide"),
    ])
    backend = ChromaFsBackend(col)
    assert isinstance(backend._cache, ContentCache)
    assert not isinstance(backend._cache, RedisContentCache)
