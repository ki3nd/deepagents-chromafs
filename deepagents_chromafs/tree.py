"""PathTree: in-memory virtual filesystem tree built from ChromaDB path metadata.

Bootstrapped once from the ``__path_tree__`` document stored in the Chroma
collection, then used for all ``ls``, ``glob``, and directory-scope operations
with zero network calls.

Path tree document format (stored as the document text of ``__path_tree__``):

    {
        "auth/oauth.md": {"isPublic": true, "groups": []},
        "auth/api-keys.mdx": {"isPublic": true, "groups": []},
        "internal/billing.md": {"isPublic": false, "groups": ["admin", "billing"]}
    }

**Slug format contract:** every key must exactly match the ``page_slug``
metadata stored on each chunk in the same collection.  Slugs may or may not
include a file extension (``auth/oauth.md``, ``Makefile``, ``Dockerfile`` are
all valid).  Extension-based glob patterns (``**/*.md``, ``**/*.{md,mdx}``)
only match slugs that carry the corresponding extension — slugs without an
extension will not match those patterns, which is the expected behavior.

The document may optionally be gzip-compressed and base64-encoded.  The loader
detects which format is present automatically.
"""

from __future__ import annotations

import base64
import gzip
import json
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import wcmatch.glob as wcglob


@dataclass(frozen=True)
class PathInfo:
    """Access-control metadata for a single path entry in the tree."""

    is_public: bool
    """Whether the path is accessible without group membership."""

    groups: frozenset[str]
    """Groups that may access this path when ``is_public`` is ``False``."""

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> PathInfo:
        """Parse a path-info dict from the path tree document.

        Args:
            data: Dict with ``isPublic`` (bool) and ``groups`` (list[str]).

        Returns:
            PathInfo instance.
        """
        raw_groups = data.get("groups", [])
        groups: list[str] = raw_groups if isinstance(raw_groups, list) else []  # type: ignore[assignment]
        return cls(
            is_public=bool(data.get("isPublic", True)),
            groups=frozenset(str(g) for g in groups),
        )


def _is_accessible(info: PathInfo, user_groups: frozenset[str]) -> bool:
    """Return True if the path is accessible to a user with the given groups.

    Args:
        info: Path access metadata.
        user_groups: Groups the current user belongs to.

    Returns:
        True when the path should be visible.
    """
    if info.is_public:
        return True
    return bool(info.groups & user_groups)


def _decompress_tree_document(raw: str) -> str:
    """Decompress a path tree document that may be gzip+base64 encoded.

    Detects format: if the string starts with ``{`` it is treated as plain
    JSON; otherwise a base64-decode + gzip-decompress is attempted.

    Args:
        raw: Raw document text from Chroma.

    Returns:
        Plain JSON string ready for ``json.loads``.

    Raises:
        ValueError: If the document cannot be decoded.
    """
    stripped = raw.strip()
    if stripped.startswith("{"):
        return stripped

    try:
        compressed = base64.b64decode(stripped)
        return gzip.decompress(compressed).decode("utf-8")
    except Exception as exc:
        msg = f"Cannot decode path tree document: {exc}"
        raise ValueError(msg) from exc


@dataclass
class PathTree:
    """In-memory virtual filesystem tree for ChromaFs.

    Virtual paths always start with ``/``.  Directory paths may optionally end
    with ``/`` when passed to methods — both forms are accepted.

    The tree is built from the path tree document stored in ChromaDB and is
    then used for all ``ls``, ``glob``, and slug-scoping operations without
    any network calls.

    Attributes:
        _file_paths: Set of all accessible virtual file paths (e.g. ``/auth/oauth.md``).
        _dir_children: Mapping of each directory to its immediate children
            (both files and subdirectories).
    """

    _file_paths: set[str] = field(default_factory=set)
    _dir_children: dict[str, list[str]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_json(
        cls,
        raw: str,
        *,
        user_groups: frozenset[str] | None = None,
    ) -> PathTree:
        """Build a PathTree from the raw path tree document string.

        Handles both plain JSON and gzip+base64 encoded documents.

        Args:
            raw: Raw text of the ``__path_tree__`` Chroma document.
            user_groups: Groups the current user belongs to.  Paths that the
                user cannot access are excluded from the tree entirely.

        Returns:
            PathTree populated with accessible paths.
        """
        json_str = _decompress_tree_document(raw)
        data: dict[str, dict[str, object]] = json.loads(json_str)

        effective_groups = user_groups or frozenset()

        file_paths: set[str] = set()
        for slug, info_dict in data.items():
            info = PathInfo.from_dict(info_dict)
            if _is_accessible(info, effective_groups):
                # Normalise: strip leading slash if present, then re-add
                clean_slug = slug.lstrip("/")
                file_paths.add("/" + clean_slug)

        return cls._build(file_paths)

    @classmethod
    def _build(cls, file_paths: set[str]) -> PathTree:
        """Construct the directory-child index from a flat set of file paths.

        Args:
            file_paths: Fully normalised virtual file paths.

        Returns:
            PathTree with ``_dir_children`` populated.
        """
        dir_children: dict[str, list[str]] = {}

        for path in file_paths:
            parts = PurePosixPath(path).parts  # ('/', 'auth', 'oauth')
            # Walk from root down to the file, registering each ancestor.
            for depth in range(1, len(parts)):
                parent = str(PurePosixPath(*parts[:depth]))
                child = str(PurePosixPath(*parts[: depth + 1]))
                children = dir_children.setdefault(parent, [])
                if child not in children:
                    children.append(child)

        # Sort children lists for deterministic output
        for children in dir_children.values():
            children.sort()

        return cls(_file_paths=file_paths, _dir_children=dir_children)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def is_file(self, path: str) -> bool:
        """Return True if the path corresponds to a file.

        Args:
            path: Virtual path (e.g. ``/auth/oauth.md``).

        Returns:
            True when path is a known file.
        """
        return path.rstrip("/") in self._file_paths

    def is_dir(self, path: str) -> bool:
        """Return True if the path corresponds to a directory.

        Args:
            path: Virtual path (e.g. ``/auth`` or ``/auth/``).

        Returns:
            True when path is a known directory.
        """
        normalised = path.rstrip("/") or "/"
        return normalised in self._dir_children

    def ls_children(self, path: str) -> list[str]:
        """Return immediate children of a directory.

        Children are returned as absolute virtual paths.  Directories are
        returned without a trailing slash; callers may add one if needed.

        Args:
            path: Virtual directory path.

        Returns:
            Sorted list of child paths.  Empty list if the directory does
            not exist or has no children.
        """
        normalised = path.rstrip("/") or "/"
        return list(self._dir_children.get(normalised, []))

    def slugs_under(self, path: str) -> list[str]:
        """Return all file paths recursively under a directory.

        Args:
            path: Virtual directory path (e.g. ``/auth`` or ``/``).

        Returns:
            Sorted list of virtual file paths under the given directory.
        """
        normalised = (path.rstrip("/") or "/") + "/"
        if normalised == "//":
            normalised = "/"

        return sorted(
            fp for fp in self._file_paths if fp.startswith(normalised) or normalised == "/"
        )

    def glob_match(self, path: str, pattern: str) -> list[str]:
        """Return file paths under ``path`` that match a glob pattern.

        Uses wcmatch for full ``**`` and brace-expansion support.

        Args:
            path: Base virtual directory path to restrict the search.
            pattern: Glob pattern relative to ``path``
                (e.g. ``**/*.md``, ``*.py``).

        Returns:
            Sorted list of matching virtual file paths.
        """
        base = path.rstrip("/") or "/"
        candidates = self.slugs_under(base)

        # Make pattern relative: strip the base prefix from candidates,
        # match against the pattern, then restore full path.
        base_prefix = "" if base == "/" else base

        matches: list[str] = []
        for fp in candidates:
            relative = fp[len(base_prefix):].lstrip("/")
            if wcglob.globmatch(relative, pattern, flags=wcglob.BRACE | wcglob.GLOBSTAR):
                matches.append(fp)

        return matches

    def path_to_slug(self, path: str) -> str:
        """Convert a virtual filesystem path to a Chroma slug.

        Strips the leading ``/`` to produce the raw slug stored in Chroma
        metadata (e.g. ``/auth/oauth.md`` → ``auth/oauth.md``).

        Args:
            path: Virtual filesystem path.

        Returns:
            Chroma slug without leading slash.
        """
        return path.lstrip("/")

    def slug_to_path(self, slug: str) -> str:
        """Convert a Chroma slug to a virtual filesystem path.

        Args:
            slug: Chroma metadata slug (e.g. ``auth/oauth.md``).

        Returns:
            Virtual path with leading slash (e.g. ``/auth/oauth.md``).
        """
        return "/" + slug.lstrip("/")
