import { beforeEach, describe, expect, it } from "vitest";
import type { Collection } from "chromadb";
import { ChromaFsBackend } from "../deepagents_chromafs/backend.js";
import { AsyncRedisContentCache, RedisContentCache, type RedisClientLike } from "../deepagents_chromafs/redis-cache.js";

class FakeRedis implements RedisClientLike {
  readonly store = new Map<string, string>();

  async get(key: string): Promise<string | null> {
    return this.store.get(key) ?? null;
  }

  async set(key: string, value: string): Promise<unknown> {
    this.store.set(key, value);
    return "OK";
  }

  async setex(key: string, _seconds: number, value: string): Promise<unknown> {
    this.store.set(key, value);
    return "OK";
  }

  async exists(...keys: string[]): Promise<number> {
    return keys.some((k) => this.store.has(k)) ? 1 : 0;
  }

  async del(...keys: string[]): Promise<number> {
    let removed = 0;
    for (const key of keys) {
      if (this.store.delete(key)) {
        removed += 1;
      }
    }
    return removed;
  }

  async *scanStream(options?: { match?: string; count?: number }): AsyncIterable<string[]> {
    const match = options?.match ?? "*";
    const prefix = match.endsWith("*") ? match.slice(0, -1) : match;
    const keys = [...this.store.keys()].filter((k) => k.startsWith(prefix));
    if (keys.length > 0) {
      yield keys;
    }
  }
}

const TREE_JSON = JSON.stringify({
  "auth/oauth.md": { isPublic: true, groups: [] },
});

function makeCollection(): Collection {
  return {
    get: async ({ ids, where }: { ids?: string[]; where?: Record<string, unknown> }) => {
      if (ids?.includes("__path_tree__")) {
        return { documents: [TREE_JSON], metadatas: [{}], ids: ["__path_tree__"] };
      }
      if (where?.["page_slug"]) {
        const slug = String(where["page_slug"]);
        if (slug === "auth/oauth.md") {
          return {
            documents: ["OAuth", "Guide"],
            metadatas: [
              { page_slug: slug, chunk_index: 0 },
              { page_slug: slug, chunk_index: 1 },
            ],
            ids: ["auth/oauth.md:0", "auth/oauth.md:1"],
          };
        }
      }
      return { documents: [], metadatas: [], ids: [] };
    },
  } as unknown as Collection;
}

describe("RedisContentCache", () => {
  let redis: FakeRedis;

  beforeEach(() => {
    redis = new FakeRedis();
  });

  it("supports async put/get/has", async () => {
    const cache = new RedisContentCache(redis, { prefix: "test" });

    await cache.putAsync("auth/oauth.md", "OAuth content");

    expect(await cache.hasAsync("auth/oauth.md")).toBe(true);
    expect(await cache.getAsync("auth/oauth.md")).toBe("OAuth content");
  });

  it("clearAsync only clears own prefix", async () => {
    const a = new RedisContentCache(redis, { prefix: "a" });
    const b = new RedisContentCache(redis, { prefix: "b" });

    await a.putAsync("auth/oauth.md", "A");
    await b.putAsync("auth/oauth.md", "B");

    await a.clearAsync();

    expect(await a.getAsync("auth/oauth.md")).toBeUndefined();
    expect(await b.getAsync("auth/oauth.md")).toBe("B");
  });
});

describe("ChromaFsBackend with Redis cache", () => {
  it("reads successfully with RedisContentCache", async () => {
    const redis = new FakeRedis();
    const cache = new RedisContentCache(redis, { prefix: "backend" });
    const backend = await ChromaFsBackend.create(makeCollection(), { cache });

    const result = await backend.read("/auth/oauth.md");

    expect(result.error).toBeUndefined();
    expect(result.content).toBe("OAuth\nGuide");
    expect(await cache.hasAsync("auth/oauth.md")).toBe(true);
  });

  it("reads successfully with AsyncRedisContentCache", async () => {
    const redis = new FakeRedis();
    const cache = new AsyncRedisContentCache(redis, { prefix: "backend_async" });
    const backend = await ChromaFsBackend.create(makeCollection(), { cache });

    const result = await backend.read("/auth/oauth.md");

    expect(result.error).toBeUndefined();
    expect(result.content).toBe("OAuth\nGuide");
    expect(await cache.hasAsync("auth/oauth.md")).toBe(true);
  });
});
