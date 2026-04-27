# deepagents-chromafs

A read-only [`BackendProtocol`](https://github.com/langchain-ai/deepagents) backend for [DeepAgents](https://github.com/langchain-ai/deepagents) that treats a [ChromaDB](https://www.trychroma.com/) collection as a virtual filesystem.

Inspired by the [ChromaFs algorithm from Mintlify](https://mintlify.com/blog/chromafs): replace expensive sandbox boot (~46 s) with an in-memory virtual filesystem bootstrapped from a single Chroma document (~100 ms).

---

## How it works

### Path tree

The entire directory tree is stored as a single JSON document in Chroma under the key `__path_tree__`:

```json
{
    "auth/oauth.md": { "isPublic": true, "groups": [] },
    "auth/api-keys.mdx": { "isPublic": true, "groups": [] },
    "internal/billing.md": { "isPublic": false, "groups": ["admin", "billing"] }
}
```

> **Slug format contract:** every key must exactly match the `page_slug` metadata on each chunk in the same collection.  Slugs may or may not carry a file extension — `auth/oauth.md`, `Makefile`, and `Dockerfile` are all valid.  Extension-based glob patterns (`**/*.md`, `**/*.py`) only match slugs that include the corresponding extension; slugs without an extension simply won't match those patterns, which is the expected behavior.

The document may optionally be gzip-compressed and base64-encoded.  On bootstrap, the backend fetches this document, applies RBAC filtering (hiding paths the user cannot access), and builds an in-memory directory index — no further network calls are needed for `ls`, `glob`, or path-scoping.

### Content (cat)

Page content is stored as chunks in Chroma, each with `page_slug` and `chunk_index` metadata fields.  On first `read`, all chunks are fetched, sorted, joined, and cached for the session lifetime.

### Grep (4-step pipeline)

1. **Scope** — derive candidate slugs from the in-memory tree (limited to the requested `path` / `glob`).
2. **Coarse filter** — Chroma `$contains` / `$regex` on `where_document` to find matching chunks.
3. **Bulk prefetch** — fetch all matched page slugs concurrently into the in-memory cache.
4. **Fine filter** — in-memory regex on cached content to produce line-level `GrepMatch` results.

### Write operations

All write operations (`write`, `edit`, `upload_files`) return an EROFS error.  The filesystem is stateless by design.

---

## Installation

```bash
pip install deepagents-chromafs
```

Or with `uv`:

```bash
uv add deepagents-chromafs
```

---

## Quick start

```python
import chromadb
from deepagents_chromafs import ChromaFsBackend

client = chromadb.Client()
collection = client.get_collection("my_docs")

backend = ChromaFsBackend(collection)

# List root directory
result = backend.ls("/")
for entry in result.entries:
    print(entry["path"], "dir" if entry.get("is_dir") else "file")

# Read a page
result = backend.read("/auth/oauth.md")
print(result.file_data["content"])

# Grep across all pages
result = backend.grep("OAuth2")
for match in result.matches:
    print(f"{match['path']}:{match['line']}: {match['text']}")

# Glob for files
result = backend.glob("**/*.md")
for entry in result.matches:
    print(entry["path"])
```

### RBAC (group-based access control)

```python
backend = ChromaFsBackend(
    collection,
    user_groups=frozenset({"admin", "billing"}),
)
```

Paths whose `isPublic` is `False` and whose `groups` list does not intersect with `user_groups` are hidden from the tree entirely — they do not appear in `ls`, `glob`, or `grep` results.

### Custom metadata field names

```python
backend = ChromaFsBackend(
    collection,
    slug_field="doc_slug",        # default: "page_slug"
    chunk_index_field="seq",      # default: "chunk_index"
)
```

---

## ChromaDB schema

Each page chunk document must have these metadata fields:

| Field | Type | Description |
|---|---|---|
| `page_slug` | `str` | Page identifier including extension (e.g. `auth/oauth.md`) |
| `chunk_index` | `int` | Chunk ordering within the page |

The path tree is stored as a single document with ID `__path_tree__`.

### Preventing `__path_tree__` from polluting search

By default ChromaDB auto-generates an embedding for every document added via
`collection.add()`, including `__path_tree__`.  This wastes embedding compute
and lets the document surface in semantic similarity searches (`collection.query()`).
Two mitigations are recommended when inserting the path tree:

**1. Zero-vector embedding (semantic search)**

Pass an explicit zero vector so the document never wins a cosine similarity
match:

```python
EMBEDDING_DIM = 1536  # match your collection's embedding dimension

collection.add(
    ids=["__path_tree__"],
    documents=[tree_json],
    embeddings=[[0.0] * EMBEDDING_DIM],
)
```

**2. Metadata marker (full-text / `where_document` queries)**

Add a metadata field that lets you exclude the document from your own queries:

```python
collection.add(
    ids=["__path_tree__"],
    documents=[tree_json],
    embeddings=[[0.0] * EMBEDDING_DIM],
    metadatas=[{"_system": True}],
)
```

Then filter it out in any custom `where_document` scan:

```python
collection.get(
    where={"_system": {"$ne": True}},
    where_document={"$contains": "access_token"},
)
```

> **Note:** `ChromaFsBackend` itself is not affected — its grep pipeline always
> scopes queries to `page_slug` metadata, so `__path_tree__` (which has no
> `page_slug`) is naturally excluded from all results.

---

## Development

```bash
# Install dev dependencies
make install

# Run tests
make test

# Lint
make lint

# Format
make format
```

---

## Algorithm reference

See the [ChromaFs algorithm post on Mintlify](https://mintlify.com/blog/chromafs) for the original description.
