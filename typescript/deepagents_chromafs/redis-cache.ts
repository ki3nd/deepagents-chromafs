/**
 * RedisContentCache: Redis-backed content cache for multi-session deployments.
 *
 * Drop-in replacement for the default in-memory ContentCache.
 * Requires ioredis: npm install ioredis
 *
 * Usage:
 *   import { RedisContentCache } from "deepagents-chromafs/redis-cache";
 *   import Redis from "ioredis";
 *
 *   const redis = new Redis({ host: "localhost", port: 6379 });
 *   const cache = new RedisContentCache(redis, { prefix: "myapp", ttl: 3600 });
 *   const backend = await ChromaFsBackend.create(collection, { cache });
 */

import { ContentCache } from "./cache.js";

export interface RedisClientLike {
  get(key: string): Promise<string | null>;
  set(key: string, value: string): Promise<unknown>;
  setex(key: string, seconds: number, value: string): Promise<unknown>;
  exists(...keys: string[]): Promise<number>;
  del(...keys: string[]): Promise<number>;
  scanStream(options?: { match?: string; count?: number }): AsyncIterable<string[]>;
}

export interface RedisContentCacheOptions {
  prefix?: string;
  ttl?: number;
}

export class RedisContentCache extends ContentCache {
  private readonly client: RedisClientLike;
  private readonly prefix: string;
  private readonly ttl: number;
  private readonly localMirror = new Map<string, string>();

  constructor(client: RedisClientLike, options: RedisContentCacheOptions = {}) {
    super();
    this.client = client;
    this.prefix = options.prefix ?? "chromafs";
    this.ttl = options.ttl ?? 3600;
  }

  private key(slug: string): string {
    return `${this.prefix}:${slug}`;
  }

  override get(slug: string): string | undefined {
    return this.localMirror.get(slug);
  }

  async getAsync(slug: string): Promise<string | undefined> {
    const raw = await this.client.get(this.key(slug));
    const value = raw ?? undefined;
    if (value !== undefined) {
      this.localMirror.set(slug, value);
    }
    return value;
  }

  override put(slug: string, content: string): void {
    this.localMirror.set(slug, content);
    void this._putAsync(slug, content);
  }

  async putAsync(slug: string, content: string): Promise<void> {
    this.localMirror.set(slug, content);
    await this._putAsync(slug, content);
  }

  private async _putAsync(slug: string, content: string): Promise<void> {
    const k = this.key(slug);
    if (this.ttl > 0) {
      await this.client.setex(k, this.ttl, content);
    } else {
      await this.client.set(k, content);
    }
  }

  override has(slug: string): boolean {
    return this.localMirror.has(slug);
  }

  async hasAsync(slug: string): Promise<boolean> {
    if (this.localMirror.has(slug)) {
      return true;
    }
    const count = await this.client.exists(this.key(slug));
    return count > 0;
  }

  override clear(): void {
    this.localMirror.clear();
    void this.clearAsync();
  }

  async clearAsync(): Promise<void> {
    this.localMirror.clear();
    const keys: string[] = [];
    for await (const batch of this.client.scanStream({ match: `${this.prefix}:*`, count: 100 })) {
      keys.push(...batch);
    }
    if (keys.length > 0) {
      await this.client.del(...keys);
    }
  }
}

/**
 * Async-first Redis cache — recommended for production use.
 * Wraps RedisContentCache and overrides the cache interface so the backend
 * can call async get/has without throwing.
 */
export class AsyncRedisContentCache extends ContentCache {
  private readonly redis: RedisContentCache;

  constructor(client: RedisClientLike, options: RedisContentCacheOptions = {}) {
    super();
    this.redis = new RedisContentCache(client, options);
  }

  override get(slug: string): string | undefined {
    return this.redis.get(slug);
  }

  override put(slug: string, content: string): void {
    this.redis.put(slug, content);
  }

  override has(slug: string): boolean {
    return this.redis.has(slug);
  }

  override clear(): void {
    this.redis.clear();
  }

  async getAsync(slug: string): Promise<string | undefined> {
    return this.redis.getAsync(slug);
  }

  async putAsync(slug: string, content: string): Promise<void> {
    await this.redis.putAsync(slug, content);
  }

  async hasAsync(slug: string): Promise<boolean> {
    return this.redis.hasAsync(slug);
  }

  async warmUp(slugs: string[]): Promise<void> {
    await Promise.all(
      slugs.map(async (slug) => {
        if (await this.redis.hasAsync(slug)) {
          await this.redis.getAsync(slug);
        }
      }),
    );
  }
}
