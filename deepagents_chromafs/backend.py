"""ChromaFsBackend: read-only BackendProtocol backed by a ChromaDB collection.

Treats a ChromaDB collection as a virtual filesystem, implementing the full
``BackendProtocol`` interface.  Writes are rejected with an EROFS error because
the filesystem is stateless and immutable — all content lives in ChromaDB.

Bootstrap is O(1) network call (fetching ``__path_tree__``); subsequent ``ls``,
``glob``, and path-scoping are pure in-memory operations.  Page content is
fetched lazily on first access and cached for the lifetime of the backend
instance.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)

from deepagents_chromafs.cache import ContentCache
from deepagents_chromafs.grep import (
    abulk_prefetch,
    bulk_prefetch,
    fetch_page,
    find_matching_slugs,
    fine_filter,
    to_chroma_filter,
)
from deepagents_chromafs.tree import PathTree

if TYPE_CHECKING:
    import chromadb

# FileOperationError literals — mirrored from protocol.py for forward compat.
# (Named constants were added after deepagents 0.5.3.)
_FILE_NOT_FOUND = "file_not_found"
_IS_DIRECTORY = "is_directory"
_PERMISSION_DENIED = "permission_denied"

# Key used to store the path tree document in Chroma
_PATH_TREE_ID = "__path_tree__"

# Returned for all write operations (this filesystem is read-only)
_EROFS_ERROR = (
    "Read-only filesystem: ChromaFsBackend does not support write operations."
)


class ChromaFsBackend(BackendProtocol):
    """Read-only ``BackendProtocol`` backed by a ChromaDB collection.

    Implements the ChromaFs algorithm from Mintlify: the entire directory
    tree is stored as a single compressed JSON document in Chroma
    (``__path_tree__``), allowing instant bootstrap (~100 ms) without
    spinning up a sandbox.

    All write-related methods (``write``, ``edit``, ``upload_files``) return
    an EROFS error.  ``download_files`` is supported and returns page content
    as UTF-8 bytes.

    Args:
        collection: A ChromaDB collection that follows the ChromaFs schema.
        user_groups: Optional set of groups for RBAC path filtering.  Paths
            whose ``groups`` list does not intersect with ``user_groups`` (and
            are not ``isPublic``) are hidden from the tree.
        slug_field: Metadata field name that stores the page slug on each
            chunk document.
        chunk_index_field: Metadata field name that stores the integer chunk
            ordering on each chunk document.
    """

    def __init__(
        self,
        collection: chromadb.Collection,
        *,
        user_groups: frozenset[str] | None = None,
        slug_field: str = "page_slug",
        chunk_index_field: str = "chunk_index",
    ) -> None:
        """Initialize the backend and bootstrap the in-memory path tree."""
        self._collection = collection
        self._slug_field = slug_field
        self._chunk_index_field = chunk_index_field
        self._cache = ContentCache()
        self._tree = self._bootstrap_tree(user_groups)

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def _bootstrap_tree(self, user_groups: frozenset[str] | None) -> PathTree:
        """Fetch ``__path_tree__`` from Chroma and build the in-memory tree.

        Args:
            user_groups: Groups for RBAC filtering; ``None`` means no filter
                (public paths only unless ``isPublic`` is True for all).

        Returns:
            Populated ``PathTree``.  Returns an empty tree when the path-tree
            document is absent (safe degraded mode — all paths missing).
        """
        results = self._collection.get(
            ids=[_PATH_TREE_ID],
            include=["documents"],
        )
        documents: list[str] = results.get("documents") or []
        if not documents or not documents[0]:
            return PathTree()
        return PathTree.from_json(documents[0], user_groups=user_groups)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_and_cache(self, slug: str) -> str:
        """Fetch a page from Chroma and populate the cache.

        Args:
            slug: Chroma page slug (e.g. ``auth/oauth.md``).

        Returns:
            Full reassembled page text.
        """
        if not self._cache.has(slug):
            content = fetch_page(
                self._collection,
                slug,
                slug_field=self._slug_field,
                chunk_index_field=self._chunk_index_field,
            )
            self._cache.put(slug, content)
        return self._cache.get(slug) or ""

    # ------------------------------------------------------------------
    # BackendProtocol — read operations
    # ------------------------------------------------------------------

    def ls(self, path: str) -> LsResult:
        """List immediate children of a virtual directory.

        Args:
            path: Absolute virtual path to list (e.g. ``/auth`` or ``/``).

        Returns:
            ``LsResult`` with ``entries`` on success, or ``error`` if the path
            does not exist or is a file.
        """
        normalised = path.rstrip("/") or "/"
        if not self._tree.is_dir(normalised):
            if self._tree.is_file(normalised):
                return LsResult(error=f"Not a directory: {path}")
            return LsResult(error=f"No such directory: {path}")

        entries: list[FileInfo] = [
            {"path": child, "is_dir": self._tree.is_dir(child)}
            for child in self._tree.ls_children(normalised)
        ]
        return LsResult(entries=entries)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Read a page from ChromaDB with optional line-range slicing.

        Chunks are reassembled in ``chunk_index`` order and the result is
        cached for subsequent calls.

        Args:
            file_path: Absolute virtual path (e.g. ``/auth/oauth.md``).
            offset: 0-indexed line to start reading from.
            limit: Maximum number of lines to return.

        Returns:
            ``ReadResult`` with ``file_data`` on success, or ``error`` if the
            path does not exist or is a directory.
        """
        normalised = file_path.rstrip("/")
        if not self._tree.is_file(normalised):
            if self._tree.is_dir(normalised):
                return ReadResult(error=f"Is a directory: {file_path}")
            return ReadResult(error=f"No such file: {file_path}")

        slug = self._tree.path_to_slug(normalised)
        content = self._fetch_and_cache(slug)

        lines = content.splitlines()
        if offset > 0 and offset >= len(lines) and lines:
            return ReadResult(
                error=f"Line offset {offset} exceeds file length ({len(lines)} lines)"
            )

        sliced = "\n".join(lines[offset : offset + limit])
        return ReadResult(file_data={"content": sliced, "encoding": "utf-8"})

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Search for a literal pattern across pages using the 4-step pipeline.

        Steps:
        1. Derive candidate slugs from the in-memory tree (scoped to ``path``
           and filtered by ``glob`` when provided).
        2. Issue a Chroma ``$contains`` coarse filter to narrow to matching
           chunks.
        3. Bulk-prefetch the matched pages into the content cache.
        4. Run an in-memory fine filter to produce line-level ``GrepMatch``
           results.

        Args:
            pattern: Literal string to search for (not regex).
            path: Optional virtual directory path to restrict the search.
            glob: Optional glob pattern to filter files by path.

        Returns:
            ``GrepResult`` with ``matches`` (possibly empty) on success.
        """
        search_root = (path or "/").rstrip("/") or "/"
        candidate_paths = self._tree.glob_match(search_root, glob) if glob else self._tree.slugs_under(search_root)
        candidate_slugs = [self._tree.path_to_slug(p) for p in candidate_paths]

        chroma_filter = to_chroma_filter(pattern, fixed_string=True)
        matched_slugs = find_matching_slugs(
            self._collection,
            chroma_filter,
            candidate_slugs,
            slug_field=self._slug_field,
        )

        if not matched_slugs:
            return GrepResult(matches=[])

        bulk_prefetch(
            self._collection,
            matched_slugs,
            self._cache,
            slug_field=self._slug_field,
            chunk_index_field=self._chunk_index_field,
        )

        matches = fine_filter(
            pattern,
            matched_slugs,
            self._cache,
            fixed_string=True,
            slug_to_path=self._tree.slug_to_path,
        )
        return GrepResult(matches=matches)

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Async grep — prefetches matched pages concurrently.

        Overrides the default ``asyncio.to_thread`` wrapper to parallelize
        the bulk-prefetch step using ``abulk_prefetch``.

        Args:
            pattern: Literal string to search for (not regex).
            path: Optional virtual directory path to restrict the search.
            glob: Optional glob pattern to filter files by path.

        Returns:
            ``GrepResult`` with ``matches`` (possibly empty) on success.
        """
        search_root = (path or "/").rstrip("/") or "/"
        candidate_paths = self._tree.glob_match(search_root, glob) if glob else self._tree.slugs_under(search_root)
        candidate_slugs = [self._tree.path_to_slug(p) for p in candidate_paths]

        chroma_filter = to_chroma_filter(pattern, fixed_string=True)
        matched_slugs = await asyncio.to_thread(
            find_matching_slugs,
            self._collection,
            chroma_filter,
            candidate_slugs,
            slug_field=self._slug_field,
        )

        if not matched_slugs:
            return GrepResult(matches=[])

        await abulk_prefetch(
            self._collection,
            matched_slugs,
            self._cache,
            slug_field=self._slug_field,
            chunk_index_field=self._chunk_index_field,
        )

        matches = fine_filter(
            pattern,
            matched_slugs,
            self._cache,
            fixed_string=True,
            slug_to_path=self._tree.slug_to_path,
        )
        return GrepResult(matches=matches)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        """Find virtual files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g. ``**/*.md``, ``*.py``).
            path: Base virtual directory path.

        Returns:
            ``GlobResult`` with matching ``FileInfo`` entries.
        """
        matched_paths = self._tree.glob_match(path, pattern)
        entries: list[FileInfo] = [{"path": p, "is_dir": False} for p in matched_paths]
        return GlobResult(matches=entries)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download page content as raw UTF-8 bytes.

        Args:
            paths: List of absolute virtual paths to download.

        Returns:
            List of ``FileDownloadResponse`` objects in input order.
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            normalised = path.rstrip("/")
            if self._tree.is_dir(normalised):
                responses.append(FileDownloadResponse(path=path, error=_IS_DIRECTORY))
                continue
            if not self._tree.is_file(normalised):
                responses.append(FileDownloadResponse(path=path, error=_FILE_NOT_FOUND))
                continue

            slug = self._tree.path_to_slug(normalised)
            content = self._fetch_and_cache(slug)
            responses.append(
                FileDownloadResponse(path=path, content=content.encode("utf-8"))
            )
        return responses

    # ------------------------------------------------------------------
    # BackendProtocol — write operations (EROFS)
    # ------------------------------------------------------------------

    def write(self, file_path: str, content: str) -> WriteResult:  # noqa: ARG002
        """Not supported — this filesystem is read-only.

        Args:
            file_path: Ignored.
            content: Ignored.

        Returns:
            ``WriteResult`` with EROFS error.
        """
        return WriteResult(error=_EROFS_ERROR)

    def edit(
        self,
        file_path: str,  # noqa: ARG002
        old_string: str,  # noqa: ARG002
        new_string: str,  # noqa: ARG002
        replace_all: bool = False,  # noqa: ARG002, FBT001, FBT002
    ) -> EditResult:
        """Not supported — this filesystem is read-only.

        Args:
            file_path: Ignored.
            old_string: Ignored.
            new_string: Ignored.
            replace_all: Ignored.

        Returns:
            ``EditResult`` with EROFS error.
        """
        return EditResult(error=_EROFS_ERROR)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Not supported — this filesystem is read-only.

        Args:
            files: Ignored.

        Returns:
            ``FileUploadResponse`` with ``permission_denied`` for every input file.
        """
        return [
            FileUploadResponse(path=path, error=_PERMISSION_DENIED)
            for path, _ in files
        ]
