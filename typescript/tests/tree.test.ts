import { describe, it, expect } from "vitest";
import { PathTree } from "../deepagents_chromafs/tree.js";

const SAMPLE_TREE_JSON = JSON.stringify({
  "auth/oauth.md": { isPublic: true, groups: [] },
  "auth/api-keys.mdx": { isPublic: true, groups: [] },
  "internal/billing.md": { isPublic: false, groups: ["admin", "billing"] },
  "public/index.md": { isPublic: true, groups: [] },
});

describe("PathTree.fromJson", () => {
  it("builds file paths from JSON", () => {
    const tree = PathTree.fromJson(SAMPLE_TREE_JSON);
    expect(tree.isFile("/auth/oauth.md")).toBe(true);
    expect(tree.isFile("/auth/api-keys.mdx")).toBe(true);
    expect(tree.isFile("/public/index.md")).toBe(true);
  });

  it("excludes private paths when no user groups provided", () => {
    const tree = PathTree.fromJson(SAMPLE_TREE_JSON);
    expect(tree.isFile("/internal/billing.md")).toBe(false);
  });

  it("includes private paths when user has matching group", () => {
    const tree = PathTree.fromJson(SAMPLE_TREE_JSON, new Set(["admin"]));
    expect(tree.isFile("/internal/billing.md")).toBe(true);
  });

  it("excludes private paths when user group does not match", () => {
    const tree = PathTree.fromJson(SAMPLE_TREE_JSON, new Set(["viewer"]));
    expect(tree.isFile("/internal/billing.md")).toBe(false);
  });
});

describe("PathTree.isDir", () => {
  it("recognises root as a directory", () => {
    const tree = PathTree.fromJson(SAMPLE_TREE_JSON);
    expect(tree.isDir("/")).toBe(true);
  });

  it("recognises intermediate dirs", () => {
    const tree = PathTree.fromJson(SAMPLE_TREE_JSON);
    expect(tree.isDir("/auth")).toBe(true);
    expect(tree.isDir("/public")).toBe(true);
  });

  it("does not treat files as dirs", () => {
    const tree = PathTree.fromJson(SAMPLE_TREE_JSON);
    expect(tree.isDir("/auth/oauth.md")).toBe(false);
  });
});

describe("PathTree.lsChildren", () => {
  it("lists root children", () => {
    const tree = PathTree.fromJson(SAMPLE_TREE_JSON);
    const children = tree.lsChildren("/");
    expect(children).toContain("/auth");
    expect(children).toContain("/public");
  });

  it("lists auth directory children", () => {
    const tree = PathTree.fromJson(SAMPLE_TREE_JSON);
    const children = tree.lsChildren("/auth");
    expect(children).toContain("/auth/oauth.md");
    expect(children).toContain("/auth/api-keys.mdx");
  });

  it("returns empty for non-existent dir", () => {
    const tree = PathTree.fromJson(SAMPLE_TREE_JSON);
    expect(tree.lsChildren("/nonexistent")).toEqual([]);
  });
});

describe("PathTree.globMatch", () => {
  it("matches **/*.md pattern", () => {
    const tree = PathTree.fromJson(SAMPLE_TREE_JSON);
    const matches = tree.globMatch("/", "**/*.md");
    expect(matches).toContain("/auth/oauth.md");
    expect(matches).toContain("/public/index.md");
    expect(matches).not.toContain("/auth/api-keys.mdx");
  });

  it("matches **/*.{md,mdx} brace pattern", () => {
    const tree = PathTree.fromJson(SAMPLE_TREE_JSON);
    const matches = tree.globMatch("/", "**/*.{md,mdx}");
    expect(matches).toContain("/auth/oauth.md");
    expect(matches).toContain("/auth/api-keys.mdx");
  });

  it("scopes to subdirectory", () => {
    const tree = PathTree.fromJson(SAMPLE_TREE_JSON);
    const matches = tree.globMatch("/auth", "**/*.md");
    expect(matches).toContain("/auth/oauth.md");
    expect(matches).not.toContain("/public/index.md");
  });
});

describe("PathTree.slugsUnder", () => {
  it("returns all files under root", () => {
    const tree = PathTree.fromJson(SAMPLE_TREE_JSON);
    const slugs = tree.slugsUnder("/");
    expect(slugs.length).toBe(3);
  });

  it("returns files under /auth only", () => {
    const tree = PathTree.fromJson(SAMPLE_TREE_JSON);
    const slugs = tree.slugsUnder("/auth");
    expect(slugs).toContain("/auth/oauth.md");
    expect(slugs).toContain("/auth/api-keys.mdx");
    expect(slugs).not.toContain("/public/index.md");
  });
});

describe("PathTree.pathToSlug / slugToPath", () => {
  it("strips leading slash", () => {
    const tree = PathTree.empty();
    expect(tree.pathToSlug("/auth/oauth.md")).toBe("auth/oauth.md");
  });

  it("adds leading slash", () => {
    const tree = PathTree.empty();
    expect(tree.slugToPath("auth/oauth.md")).toBe("/auth/oauth.md");
  });
});
