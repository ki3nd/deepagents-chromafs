"""Unit tests for ChromaFsBackend."""

from __future__ import annotations

import json

from deepagents_chromafs.backend import (
    _FILE_NOT_FOUND,
    _IS_DIRECTORY,
    _PERMISSION_DENIED,
    ChromaFsBackend,
)
from tests.unit_tests.fake_collection import FakeCollection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TREE_DATA = {
    "auth/oauth": {"isPublic": True, "groups": []},
    "auth/api-keys": {"isPublic": True, "groups": []},
    "docs/intro": {"isPublic": True, "groups": []},
    "internal/billing": {"isPublic": False, "groups": ["admin"]},
}


def _make_tree_doc() -> dict:
    return {"id": "__path_tree__", "document": json.dumps(_TREE_DATA)}


def _make_chunk(slug: str, chunk_index: int, text: str) -> dict:
    return {
        "id": f"{slug}:{chunk_index}",
        "document": text,
        "page_slug": slug,
        "chunk_index": chunk_index,
    }


def _build_collection(*extra_docs: dict) -> FakeCollection:
    docs = [_make_tree_doc(), *extra_docs]
    return FakeCollection(docs)


def _backend(**kwargs: object) -> ChromaFsBackend:
    collection = _build_collection(
        _make_chunk("auth/oauth", 0, "OAuth2 guide line one\nOAuth2 guide line two"),
        _make_chunk("auth/api-keys", 0, "API keys section one"),
        _make_chunk("auth/api-keys", 1, "API keys section two"),
        _make_chunk("docs/intro", 0, "Welcome to the docs\nGetting started here"),
    )
    return ChromaFsBackend(collection, **kwargs)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_empty_collection():
    backend = ChromaFsBackend(FakeCollection([]))
    result = backend.ls("/")
    # Empty tree — no such directory
    assert result.error is not None


def test_bootstrap_builds_tree():
    backend = _backend()
    result = backend.ls("/auth")
    assert result.error is None
    paths = {e["path"] for e in result.entries}
    assert "/auth/oauth" in paths
    assert "/auth/api-keys" in paths


def test_bootstrap_rbac_hides_private():
    backend = _backend()
    # /internal/billing is hidden without group membership
    result = backend.ls("/internal")
    assert result.error is not None


def test_bootstrap_rbac_with_group():
    collection = _build_collection(_make_chunk("internal/billing", 0, "billing text"))
    backend = ChromaFsBackend(collection, user_groups=frozenset({"admin"}))
    result = backend.ls("/internal")
    assert result.error is None
    paths = {e["path"] for e in result.entries}
    assert "/internal/billing" in paths


# ---------------------------------------------------------------------------
# ls
# ---------------------------------------------------------------------------


def test_ls_root():
    backend = _backend()
    result = backend.ls("/")
    assert result.error is None
    paths = {e["path"] for e in result.entries}
    assert "/auth" in paths
    assert "/docs" in paths


def test_ls_subdirectory():
    backend = _backend()
    result = backend.ls("/auth")
    assert result.error is None
    paths = {e["path"] for e in result.entries}
    assert "/auth/oauth" in paths
    assert "/auth/api-keys" in paths
    assert "/docs/intro" not in paths


def test_ls_not_found():
    backend = _backend()
    result = backend.ls("/nonexistent")
    assert result.error is not None
    assert result.entries is None


def test_ls_file_as_dir():
    backend = _backend()
    result = backend.ls("/auth/oauth")
    assert result.error is not None
    assert "directory" in result.error.lower()


def test_ls_entries_mark_dirs():
    backend = _backend()
    result = backend.ls("/")
    by_path = {e["path"]: e for e in result.entries}
    assert by_path["/auth"]["is_dir"] is True


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


def test_read_file():
    backend = _backend()
    result = backend.read("/auth/oauth")
    assert result.error is None
    assert result.file_data is not None
    assert "OAuth2" in result.file_data["content"]
    assert result.file_data["encoding"] == "utf-8"


def test_read_file_offset():
    backend = _backend()
    result = backend.read("/auth/oauth", offset=1)
    assert result.error is None
    # offset=1 skips the first line
    assert "OAuth2 guide line two" in result.file_data["content"]
    assert "OAuth2 guide line one" not in result.file_data["content"]


def test_read_file_limit():
    backend = _backend()
    result = backend.read("/auth/oauth", limit=1)
    assert result.error is None
    lines = result.file_data["content"].splitlines()
    assert len(lines) == 1
    assert "line one" in lines[0]


def test_read_offset_exceeds_length():
    backend = _backend()
    result = backend.read("/auth/oauth", offset=999)
    assert result.error is not None
    assert "offset" in result.error.lower()


def test_read_not_found():
    backend = _backend()
    result = backend.read("/nonexistent/file")
    assert result.error is not None
    assert result.file_data is None


def test_read_directory_as_file():
    backend = _backend()
    result = backend.read("/auth")
    assert result.error is not None
    assert "directory" in result.error.lower()


def test_read_caches_content():
    backend = _backend()
    backend.read("/auth/oauth")
    # Second read should hit the cache (no Chroma call needed — cache is populated)
    assert backend._cache.has("auth/oauth")


def test_read_multi_chunk_joined():
    backend = _backend()
    result = backend.read("/auth/api-keys")
    assert result.error is None
    content = result.file_data["content"]
    assert "section one" in content
    assert "section two" in content


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


def test_grep_finds_pattern():
    backend = _backend()
    result = backend.grep("OAuth2")
    assert result.error is None
    assert result.matches is not None
    paths = {m["path"] for m in result.matches}
    assert "/auth/oauth" in paths


def test_grep_no_matches():
    backend = _backend()
    result = backend.grep("XYZZY_NOT_PRESENT")
    assert result.error is None
    assert result.matches == []


def test_grep_scoped_to_path():
    backend = _backend()
    result = backend.grep("guide", path="/auth/oauth")
    # Only auth/oauth should match; verify docs/intro is excluded
    paths = {m["path"] for m in result.matches}
    assert "/docs/intro" not in paths


def test_grep_with_glob():
    backend = _backend()
    # Search only in docs/** — OAuth2 is only in auth/
    result = backend.grep("OAuth2", glob="**")
    # glob="**" matches everything, just verifying no crash
    assert result.error is None


def test_grep_line_number():
    backend = _backend()
    result = backend.grep("line two")
    matches = [m for m in result.matches if m["path"] == "/auth/oauth"]
    assert len(matches) == 1
    assert matches[0]["line"] == 2  # 1-indexed


# ---------------------------------------------------------------------------
# glob
# ---------------------------------------------------------------------------


def test_glob_star():
    backend = _backend()
    result = backend.glob("*", path="/auth")
    assert result.error is None
    paths = {e["path"] for e in result.matches}
    assert "/auth/oauth" in paths
    assert "/auth/api-keys" in paths


def test_glob_double_star():
    backend = _backend()
    result = backend.glob("**", path="/")
    paths = {e["path"] for e in result.matches}
    assert "/auth/oauth" in paths
    assert "/docs/intro" in paths


def test_glob_no_matches():
    backend = _backend()
    result = backend.glob("*.py", path="/")
    assert result.error is None
    assert result.matches == []


# ---------------------------------------------------------------------------
# glob — slugs with file extensions
# ---------------------------------------------------------------------------

_TREE_DATA_EXT = {
    "auth/oauth.md": {"isPublic": True, "groups": []},
    "auth/api-keys.mdx": {"isPublic": True, "groups": []},
    "docs/intro.md": {"isPublic": True, "groups": []},
    "docs/reference.mdx": {"isPublic": True, "groups": []},
    "scripts/build.py": {"isPublic": True, "groups": []},
}


def _backend_with_ext(**kwargs: object) -> ChromaFsBackend:
    """Backend whose slugs carry file extensions (real-world format)."""
    col = FakeCollection([
        {"id": "__path_tree__", "document": json.dumps(_TREE_DATA_EXT)},
        _make_chunk("auth/oauth.md", 0, "OAuth2 guide\nOAuth2 details"),
        _make_chunk("auth/api-keys.mdx", 0, "API keys overview"),
        _make_chunk("docs/intro.md", 0, "Welcome to docs\nGetting started"),
        _make_chunk("docs/reference.mdx", 0, "Full reference"),
        _make_chunk("scripts/build.py", 0, "#!/usr/bin/env python\nprint('build')"),
    ])
    return ChromaFsBackend(col, **kwargs)


def test_glob_md_extension():
    """**/*.md returns only .md slugs."""
    backend = _backend_with_ext()
    result = backend.glob("**/*.md", path="/")
    assert result.error is None
    paths = {e["path"] for e in result.matches}
    assert "/auth/oauth.md" in paths
    assert "/docs/intro.md" in paths
    assert "/auth/api-keys.mdx" not in paths
    assert "/scripts/build.py" not in paths


def test_glob_mdx_extension():
    """**/*.mdx returns only .mdx slugs."""
    backend = _backend_with_ext()
    result = backend.glob("**/*.mdx", path="/")
    assert result.error is None
    paths = {e["path"] for e in result.matches}
    assert "/auth/api-keys.mdx" in paths
    assert "/docs/reference.mdx" in paths
    assert "/auth/oauth.md" not in paths
    assert "/scripts/build.py" not in paths


def test_glob_brace_extension():
    """**/*.{md,mdx} matches both .md and .mdx files."""
    backend = _backend_with_ext()
    result = backend.glob("**/*.{md,mdx}", path="/")
    assert result.error is None
    paths = {e["path"] for e in result.matches}
    assert "/auth/oauth.md" in paths
    assert "/auth/api-keys.mdx" in paths
    assert "/docs/intro.md" in paths
    assert "/docs/reference.mdx" in paths
    assert "/scripts/build.py" not in paths


def test_glob_py_extension_with_ext_backend():
    """**/*.py — non-vacuous: tree actually contains a .py file."""
    backend = _backend_with_ext()
    result = backend.glob("**/*.py", path="/")
    assert result.error is None
    paths = {e["path"] for e in result.matches}
    assert "/scripts/build.py" in paths
    assert "/auth/oauth.md" not in paths


def test_glob_star_with_ext_files():
    """* in /auth matches files with extensions without filtering by extension."""
    backend = _backend_with_ext()
    result = backend.glob("*", path="/auth")
    assert result.error is None
    paths = {e["path"] for e in result.matches}
    assert "/auth/oauth.md" in paths
    assert "/auth/api-keys.mdx" in paths
    assert "/docs/intro.md" not in paths


def test_glob_specific_file_with_ext():
    """Exact filename pattern oauth.md matches only that one file."""
    backend = _backend_with_ext()
    result = backend.glob("oauth.md", path="/auth")
    assert result.error is None
    paths = {e["path"] for e in result.matches}
    assert paths == {"/auth/oauth.md"}


# ---------------------------------------------------------------------------
# write / edit / upload_files (EROFS)
# ---------------------------------------------------------------------------


def test_write_returns_erofs():
    backend = _backend()
    result = backend.write("/auth/new", "content")
    assert result.error is not None
    assert result.path is None


def test_edit_returns_erofs():
    backend = _backend()
    result = backend.edit("/auth/oauth", "old", "new")
    assert result.error is not None


def test_upload_files_returns_permission_denied():
    backend = _backend()
    responses = backend.upload_files([("/auth/new", b"data")])
    assert len(responses) == 1
    assert responses[0].error == _PERMISSION_DENIED
    assert responses[0].path == "/auth/new"


# ---------------------------------------------------------------------------
# download_files
# ---------------------------------------------------------------------------


def test_download_files_success():
    backend = _backend()
    responses = backend.download_files(["/auth/oauth"])
    assert len(responses) == 1
    resp = responses[0]
    assert resp.error is None
    assert resp.content is not None
    assert b"OAuth2" in resp.content


def test_download_files_not_found():
    backend = _backend()
    responses = backend.download_files(["/nonexistent"])
    assert responses[0].error == _FILE_NOT_FOUND
    assert responses[0].content is None


def test_download_files_directory():
    backend = _backend()
    responses = backend.download_files(["/auth"])
    assert responses[0].error == _IS_DIRECTORY


def test_download_files_mixed():
    backend = _backend()
    responses = backend.download_files(["/auth/oauth", "/missing"])
    assert responses[0].error is None
    assert responses[0].content is not None
    assert responses[1].error == "file_not_found"


def test_download_files_returns_utf8_bytes():
    backend = _backend()
    responses = backend.download_files(["/docs/intro"])
    assert isinstance(responses[0].content, bytes)
    # Should decode cleanly as UTF-8
    decoded = responses[0].content.decode("utf-8")
    assert "Welcome" in decoded


# ---------------------------------------------------------------------------
# async methods
# ---------------------------------------------------------------------------


async def test_als():
    backend = _backend()
    result = await backend.als("/auth")
    assert result.error is None
    paths = {e["path"] for e in result.entries}
    assert "/auth/oauth" in paths


async def test_aread():
    backend = _backend()
    result = await backend.aread("/auth/oauth")
    assert result.error is None
    assert "OAuth2" in result.file_data["content"]


async def test_agrep():
    backend = _backend()
    result = await backend.agrep("OAuth2")
    assert result.error is None
    paths = {m["path"] for m in result.matches}
    assert "/auth/oauth" in paths


async def test_aglob():
    backend = _backend()
    result = await backend.aglob("*", path="/auth")
    assert result.error is None
    paths = {e["path"] for e in result.matches}
    assert "/auth/oauth" in paths


async def test_adownload_files():
    backend = _backend()
    responses = await backend.adownload_files(["/auth/oauth"])
    assert responses[0].error is None
    assert b"OAuth2" in responses[0].content
