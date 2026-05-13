# deepagents-chromafs

A read-only [`BackendProtocolV2`](https://github.com/ki3nd/deepagents) backend for [DeepAgents](https://github.com/ki3nd/deepagents) (JS/TS) that treats a [ChromaDB](https://www.trychroma.com/) collection as a virtual filesystem.

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

> **Slug format contract:** every key must exactly match the `page_slug` metadata on each chunk in the same collection. Slugs may or may not carry a file extension — `auth/oauth.md`, `Makefile`, and `Dockerfile` are all valid. Extension-based glob patterns (`**/*.md`, `**/*.py`) only match slugs that include the corresponding extension.

The document may optionally be gzip-compressed and base64-encoded. On bootstrap, the backend fetches this document, applies RBAC filtering, and builds an in-memory directory index — no further network calls are needed for `ls`, `glob`, or path-scoping.

### Content (read)

Page content is stored as chunks in Chroma, each with `page_slug` and `chunk_index` metadata fields. On first `read`, all chunks are fetched, sorted, joined, and cached for the session lifetime.

### Grep (4-step pipeline)

1. **Scope** — derive candidate slugs from the in-memory tree (limited to the requested `path` / `glob`).
2. **Coarse filter** — Chroma `$contains` / `$regex` on `where_document` to find matching chunks.
3. **Bulk prefetch** — fetch all matched page slugs concurrently into the in-memory cache.
4. **Fine filter** — in-memory regex on cached content to produce line-level `GrepMatch` results.

### Write operations

All write operations return an `EROFS` error. The filesystem is stateless by design.

---

## Installation

```bash
npm install deepagents-chromafs
```

With Redis cache support:

```bash
npm install deepagents-chromafs ioredis
```

---

## Quick start

```typescript
import chromadb from "chromadb";
import { ChromaFsBackend } from "deepagents-chromafs";

const client = new ChromaClient();
const collection = await client.getCollection({ name: "my_docs" });

const backend = await ChromaFsBackend.create(collection);

// List root directory
const ls = await backend.ls("/");
for (const entry of ls.entries) {
  console.log(entry.path, entry.is_dir ? "dir" : "file");
}

// Read a page
const read = await backend.read("/auth/oauth.md");
console.log(read.content);

// Grep across all pages
const grep = await backend.grep("OAuth2");
for (const match of grep.matches) {
  console.log(`${match.path}:${match.line}: ${match.text}`);
}

// Glob for files
const glob = await backend.glob("**/*.md");
for (const entry of glob.matches) {
  console.log(entry.path);
}
```

### RBAC (group-based access control)

```typescript
const backend = await ChromaFsBackend.create(collection, {
  userGroups: new Set(["admin", "billing"]),
});
```

Paths whose `isPublic` is `false` and whose `groups` list does not intersect with `userGroups` are hidden from the tree entirely — they do not appear in `ls`, `glob`, or `grep` results.

### Custom metadata field names

```typescript
const backend = await ChromaFsBackend.create(collection, {
  slugField: "doc_slug",        // default: "page_slug"
  chunkIndexField: "seq",       // default: "chunk_index"
});
```

### Redis cache (multi-session / multi-worker)

By default, page content is cached in-memory for the lifetime of the `ChromaFsBackend` instance. For multi-session or multi-worker deployments, plug in `RedisContentCache` to share the cache across processes:

```typescript
import { Redis } from "ioredis";
import { ChromaFsBackend } from "deepagents-chromafs";
import { RedisContentCache } from "deepagents-chromafs/redis-cache";

const cache = new RedisContentCache(
  new Redis({ host: "localhost", port: 6379 }),
  {
    prefix: "myapp",  // namespace — avoids key collisions between collections
    ttl: 3600,        // seconds; 0 = no expiry
  },
);

const backend = await ChromaFsBackend.create(collection, { cache });
```

Any `ContentCache` subclass is accepted, so you can wire in other backends by subclassing `ContentCache` and overriding `get`, `put`, `has`, and `clear`.

---

## ChromaDB schema

Each page chunk document must have these metadata fields:

| Field | Type | Description |
|---|---|---|
| `page_slug` | `string` | Page identifier including extension (e.g. `auth/oauth.md`) |
| `chunk_index` | `number` | Chunk ordering within the page |

The path tree is stored as a single document with ID `__path_tree__`.

### Preventing `__path_tree__` from polluting search

Pass an explicit zero vector so the document never wins a cosine similarity match:

```typescript
const EMBEDDING_DIM = 1536; // match your collection's embedding dimension

await collection.add({
  ids: ["__path_tree__"],
  documents: [treeJson],
  embeddings: [new Array(EMBEDDING_DIM).fill(0)],
  metadatas: [{ _system: true }],
});
```

---

## Development

```bash
# Install dependencies
npm install

# Run tests
npm test

# Type-check
npm run lint

# Build
npm run build
```

---

## Algorithm reference

See the [ChromaFs algorithm post on Mintlify](https://mintlify.com/blog/chromafs) for the original description.
