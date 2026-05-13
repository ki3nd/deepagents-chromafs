import { describe, it, expect, beforeEach } from "vitest";
import { ContentCache } from "../deepagents_chromafs/cache.js";
import { toChromaFilter, fineFilter, parseGroupsMetadata } from "../deepagents_chromafs/grep.js";

describe("toChromaFilter", () => {
  it("returns $contains for fixed string (default)", () => {
    expect(toChromaFilter("OAuth2")).toEqual({ $contains: "OAuth2" });
  });

  it("returns $regex for case-insensitive fixed string", () => {
    const filter = toChromaFilter("OAuth2", { ignoreCase: true });
    expect(filter).toEqual({ $regex: "(?i)OAuth2" });
  });

  it("returns $regex for non-fixed string", () => {
    const filter = toChromaFilter("auth.*token", { fixedString: false });
    expect(filter).toEqual({ $regex: "auth.*token" });
  });

  it("returns case-insensitive $regex for non-fixed string with ignoreCase", () => {
    const filter = toChromaFilter("auth.*token", { fixedString: false, ignoreCase: true });
    expect(filter).toEqual({ $regex: "(?i)auth.*token" });
  });
});

describe("parseGroupsMetadata", () => {
  it("parses JSON array string", () => {
    expect(parseGroupsMetadata('["admin", "billing"]')).toEqual(["admin", "billing"]);
  });

  it("passes through array values", () => {
    expect(parseGroupsMetadata(["admin"])).toEqual(["admin"]);
  });

  it("returns empty array for null", () => {
    expect(parseGroupsMetadata(null)).toEqual([]);
  });

  it("returns empty array for invalid JSON", () => {
    expect(parseGroupsMetadata("not json")).toEqual([]);
  });
});

describe("fineFilter", () => {
  let cache: ContentCache;

  beforeEach(() => {
    cache = new ContentCache();
    cache.put("auth/oauth.md", "# OAuth2\nThis uses OAuth2 token\nSome other line\n");
    cache.put("public/index.md", "# Welcome\nNo auth content here\n");
  });

  it("finds matching lines", () => {
    const matches = fineFilter("OAuth2", ["auth/oauth.md"], cache);
    expect(matches).toHaveLength(2);
    expect(matches[0]).toMatchObject({ path: "/auth/oauth.md", line: 1, text: "# OAuth2" });
    expect(matches[1]).toMatchObject({ path: "/auth/oauth.md", line: 2 });
  });

  it("returns empty array when pattern not found", () => {
    const matches = fineFilter("nonexistent", ["auth/oauth.md"], cache);
    expect(matches).toHaveLength(0);
  });

  it("case-insensitive matching", () => {
    const matches = fineFilter("oauth2", ["auth/oauth.md"], cache, { ignoreCase: true });
    expect(matches.length).toBeGreaterThan(0);
  });

  it("regex matching (non-fixed)", () => {
    const matches = fineFilter("OAuth\\d", ["auth/oauth.md"], cache, { fixedString: false });
    expect(matches.length).toBeGreaterThan(0);
  });

  it("uses custom slugToPath resolver", () => {
    const matches = fineFilter("OAuth2", ["auth/oauth.md"], cache, {
      slugToPath: (slug) => `/custom/${slug}`,
    });
    expect(matches[0]?.path).toBe("/custom/auth/oauth.md");
  });

  it("skips slugs not in cache", () => {
    const matches = fineFilter("OAuth2", ["missing/page.md"], cache);
    expect(matches).toHaveLength(0);
  });

  it("returns empty for invalid regex", () => {
    const matches = fineFilter("[invalid", ["auth/oauth.md"], cache, { fixedString: false });
    expect(matches).toHaveLength(0);
  });
});
