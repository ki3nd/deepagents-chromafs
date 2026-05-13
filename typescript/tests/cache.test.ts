import { describe, it, expect, beforeEach } from "vitest";
import { ContentCache } from "../deepagents_chromafs/cache.js";

describe("ContentCache", () => {
  let cache: ContentCache;

  beforeEach(() => {
    cache = new ContentCache();
  });

  it("returns undefined for missing slug", () => {
    expect(cache.get("auth/oauth.md")).toBeUndefined();
  });

  it("stores and retrieves content", () => {
    cache.put("auth/oauth.md", "# OAuth\nContent here");
    expect(cache.get("auth/oauth.md")).toBe("# OAuth\nContent here");
  });

  it("has() returns false for missing slug", () => {
    expect(cache.has("auth/oauth.md")).toBe(false);
  });

  it("has() returns true after put", () => {
    cache.put("auth/oauth.md", "content");
    expect(cache.has("auth/oauth.md")).toBe(true);
  });

  it("clear() removes all entries", () => {
    cache.put("a", "content a");
    cache.put("b", "content b");
    cache.clear();
    expect(cache.size).toBe(0);
    expect(cache.has("a")).toBe(false);
  });

  it("size reflects number of entries", () => {
    expect(cache.size).toBe(0);
    cache.put("a", "x");
    cache.put("b", "y");
    expect(cache.size).toBe(2);
  });
});
