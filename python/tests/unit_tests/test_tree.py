"""Unit tests for PathTree."""

from __future__ import annotations

import base64
import gzip
import json

import pytest

from deepagents_chromafs.tree import PathTree, _decompress_tree_document

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TREE_DATA = {
    "auth/oauth": {"isPublic": True, "groups": []},
    "auth/api-keys": {"isPublic": True, "groups": []},
    "internal/billing": {"isPublic": False, "groups": ["admin", "billing"]},
    "internal/reports": {"isPublic": False, "groups": ["admin"]},
    "public/home": {"isPublic": True, "groups": []},
}

_TREE_JSON = json.dumps(_TREE_DATA)


def _gz_b64(data: str) -> str:
    """Return gzip+base64 encoded version of a string."""
    compressed = gzip.compress(data.encode("utf-8"))
    return base64.b64encode(compressed).decode("ascii")


# ---------------------------------------------------------------------------
# _decompress_tree_document
# ---------------------------------------------------------------------------


def test_decompress_plain_json():
    assert _decompress_tree_document(_TREE_JSON) == _TREE_JSON


def test_decompress_gzip_b64():
    encoded = _gz_b64(_TREE_JSON)
    assert json.loads(_decompress_tree_document(encoded)) == _TREE_DATA


def test_decompress_invalid_raises():
    with pytest.raises(ValueError, match="Cannot decode"):
        _decompress_tree_document("not-valid-base64!!!")


# ---------------------------------------------------------------------------
# PathTree.from_json — no RBAC (public only behaviour)
# ---------------------------------------------------------------------------


def test_from_json_all_public_paths_visible():
    tree = PathTree.from_json(_TREE_JSON, user_groups=None)
    # Public paths
    assert tree.is_file("/auth/oauth")
    assert tree.is_file("/auth/api-keys")
    assert tree.is_file("/public/home")
    # Private paths — no user_groups means no group membership → hidden
    assert not tree.is_file("/internal/billing")
    assert not tree.is_file("/internal/reports")


def test_from_json_with_matching_group():
    tree = PathTree.from_json(_TREE_JSON, user_groups=frozenset({"billing"}))
    assert tree.is_file("/internal/billing")
    assert not tree.is_file("/internal/reports")


def test_from_json_with_admin_group():
    tree = PathTree.from_json(_TREE_JSON, user_groups=frozenset({"admin"}))
    assert tree.is_file("/internal/billing")
    assert tree.is_file("/internal/reports")


def test_from_json_compressed():
    tree = PathTree.from_json(_gz_b64(_TREE_JSON))
    assert tree.is_file("/auth/oauth")
    assert tree.is_file("/public/home")


# ---------------------------------------------------------------------------
# PathTree — is_file / is_dir
# ---------------------------------------------------------------------------


def test_is_file_true():
    tree = PathTree.from_json(_TREE_JSON)
    assert tree.is_file("/auth/oauth")


def test_is_file_false_for_directory():
    tree = PathTree.from_json(_TREE_JSON)
    assert not tree.is_file("/auth")


def test_is_dir_true():
    tree = PathTree.from_json(_TREE_JSON)
    assert tree.is_dir("/auth")
    assert tree.is_dir("/")


def test_is_dir_false_for_file():
    tree = PathTree.from_json(_TREE_JSON)
    assert not tree.is_dir("/auth/oauth")


def test_trailing_slash_normalised():
    tree = PathTree.from_json(_TREE_JSON)
    assert tree.is_dir("/auth/")
    # File paths should also tolerate trailing slash
    assert tree.is_file("/auth/oauth/")


# ---------------------------------------------------------------------------
# PathTree — ls_children
# ---------------------------------------------------------------------------


def test_ls_children_root():
    tree = PathTree.from_json(_TREE_JSON)
    children = tree.ls_children("/")
    # Root should have auth and public (internal has no visible children without group)
    assert "/auth" in children
    assert "/public" in children


def test_ls_children_auth():
    tree = PathTree.from_json(_TREE_JSON)
    children = tree.ls_children("/auth")
    assert "/auth/oauth" in children
    assert "/auth/api-keys" in children


def test_ls_children_missing_dir():
    tree = PathTree.from_json(_TREE_JSON)
    assert tree.ls_children("/nonexistent") == []


# ---------------------------------------------------------------------------
# PathTree — slugs_under
# ---------------------------------------------------------------------------


def test_slugs_under_root():
    tree = PathTree.from_json(_TREE_JSON)
    slugs = tree.slugs_under("/")
    assert "/auth/oauth" in slugs
    assert "/auth/api-keys" in slugs
    assert "/public/home" in slugs


def test_slugs_under_subdirectory():
    tree = PathTree.from_json(_TREE_JSON)
    slugs = tree.slugs_under("/auth")
    assert "/auth/oauth" in slugs
    assert "/auth/api-keys" in slugs
    assert "/public/home" not in slugs


def test_slugs_under_empty_for_unknown():
    tree = PathTree.from_json(_TREE_JSON)
    assert tree.slugs_under("/nope") == []


# ---------------------------------------------------------------------------
# PathTree — glob_match (slugs without extension)
# ---------------------------------------------------------------------------


def test_glob_match_star():
    tree = PathTree.from_json(_TREE_JSON)
    # Matches only direct children of /auth
    matches = tree.glob_match("/auth", "*")
    assert "/auth/oauth" in matches
    assert "/auth/api-keys" in matches
    assert "/public/home" not in matches


def test_glob_match_double_star():
    tree = PathTree.from_json(_TREE_JSON)
    matches = tree.glob_match("/", "**/*.py")
    # No .py files in our tree
    assert matches == []


def test_glob_match_from_root():
    tree = PathTree.from_json(_TREE_JSON)
    matches = tree.glob_match("/", "auth/*")
    assert "/auth/oauth" in matches
    assert "/auth/api-keys" in matches
    assert "/public/home" not in matches


# ---------------------------------------------------------------------------
# PathTree — glob_match with file extensions
# ---------------------------------------------------------------------------

_TREE_DATA_EXT = {
    "auth/oauth.md": {"isPublic": True, "groups": []},
    "auth/api-keys.mdx": {"isPublic": True, "groups": []},
    "docs/intro.md": {"isPublic": True, "groups": []},
    "docs/reference.mdx": {"isPublic": True, "groups": []},
    "scripts/build.py": {"isPublic": True, "groups": []},
    "scripts/deploy.sh": {"isPublic": True, "groups": []},
    "internal/billing.md": {"isPublic": False, "groups": ["admin"]},
}

_TREE_JSON_EXT = json.dumps(_TREE_DATA_EXT)


def test_glob_match_md_extension():
    """**/*.md only returns .md files, ignoring .mdx, .py, .sh."""
    tree = PathTree.from_json(_TREE_JSON_EXT)
    matches = tree.glob_match("/", "**/*.md")
    assert "/auth/oauth.md" in matches
    assert "/docs/intro.md" in matches
    assert "/auth/api-keys.mdx" not in matches
    assert "/scripts/build.py" not in matches
    assert "/scripts/deploy.sh" not in matches
    # Private file excluded by RBAC (no user_groups)
    assert "/internal/billing.md" not in matches


def test_glob_match_mdx_extension():
    """**/*.mdx only returns .mdx files."""
    tree = PathTree.from_json(_TREE_JSON_EXT)
    matches = tree.glob_match("/", "**/*.mdx")
    assert "/auth/api-keys.mdx" in matches
    assert "/docs/reference.mdx" in matches
    assert "/auth/oauth.md" not in matches
    assert "/scripts/build.py" not in matches


def test_glob_match_brace_extension():
    """**/*.{md,mdx} matches both .md and .mdx but not other extensions."""
    tree = PathTree.from_json(_TREE_JSON_EXT)
    matches = tree.glob_match("/", "**/*.{md,mdx}")
    assert "/auth/oauth.md" in matches
    assert "/auth/api-keys.mdx" in matches
    assert "/docs/intro.md" in matches
    assert "/docs/reference.mdx" in matches
    assert "/scripts/build.py" not in matches
    assert "/scripts/deploy.sh" not in matches


def test_glob_match_py_extension():
    """**/*.py returns only .py files — non-vacuous extension test."""
    tree = PathTree.from_json(_TREE_JSON_EXT)
    matches = tree.glob_match("/", "**/*.py")
    assert "/scripts/build.py" in matches
    assert "/auth/oauth.md" not in matches
    assert "/scripts/deploy.sh" not in matches


def test_glob_match_star_includes_ext_files():
    """* (single star) in /auth still matches files that have extensions."""
    tree = PathTree.from_json(_TREE_JSON_EXT)
    matches = tree.glob_match("/auth", "*")
    assert "/auth/oauth.md" in matches
    assert "/auth/api-keys.mdx" in matches
    # Other directories must not appear
    assert "/docs/intro.md" not in matches


def test_glob_match_specific_file_with_ext():
    """Exact filename pattern oauth.md matches only that file."""
    tree = PathTree.from_json(_TREE_JSON_EXT)
    matches = tree.glob_match("/auth", "oauth.md")
    assert matches == ["/auth/oauth.md"]


def test_glob_match_subdir_with_ext():
    """docs/* returns all files under /docs regardless of extension."""
    tree = PathTree.from_json(_TREE_JSON_EXT)
    matches = tree.glob_match("/", "docs/*")
    assert "/docs/intro.md" in matches
    assert "/docs/reference.mdx" in matches
    assert "/auth/oauth.md" not in matches


def test_glob_match_double_star_md_with_admin():
    """RBAC: private .md file appears when user has the required group."""
    tree = PathTree.from_json(_TREE_JSON_EXT, user_groups=frozenset({"admin"}))
    matches = tree.glob_match("/", "**/*.md")
    assert "/internal/billing.md" in matches


# ---------------------------------------------------------------------------
# PathTree — path_to_slug / slug_to_path
# ---------------------------------------------------------------------------


def test_path_to_slug():
    tree = PathTree.from_json(_TREE_JSON)
    assert tree.path_to_slug("/auth/oauth") == "auth/oauth"
    assert tree.path_to_slug("auth/oauth") == "auth/oauth"


def test_slug_to_path():
    tree = PathTree.from_json(_TREE_JSON)
    assert tree.slug_to_path("auth/oauth") == "/auth/oauth"
    assert tree.slug_to_path("/auth/oauth") == "/auth/oauth"
