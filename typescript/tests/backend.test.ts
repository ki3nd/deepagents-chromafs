import { describe, it, expect, beforeEach, vi } from "vitest";
import { ChromaFsBackend } from "../deepagents_chromafs/backend.js";
import type { Collection } from "chromadb";

const TREE_JSON = JSON.stringify({
  "auth/oauth.md": { isPublic: true, groups: [] },
  "auth/api-keys.mdx": { isPublic: true, groups: [] },
  "internal/billing.md": { isPublic: false, groups: ["admin"] },
  "public/index.md": { isPublic: true, groups: [] },
});

const PAGE_CONTENT: Record<string, string> = {
  "auth/oauth.md": "# OAuth2\nThis uses OAuth2 token\nSome other line",
  "auth/api-keys.mdx": "# API Keys\nManage your keys here",
  "public/index.md": "# Welcome\nWelcome to the docs",
};

function makeCollection(): Collection {
  const mock = {
    get: vi.fn(async ({ ids, where }: { ids?: string[]; where?: Record<string, unknown> }) => {
      if (ids?.includes("__path_tree__")) {
        return { documents: [TREE_JSON], metadatas: [{}], ids: ["__path_tree__"] };
      }
      if (where?.["page_slug"]) {
        const slug = String(where["page_slug"]);
        const content = PAGE_CONTENT[slug];
        if (!content) return { documents: [], metadatas: [], ids: [] };
        const chunks = content.split("\n");
        return {
          documents: chunks,
          metadatas: chunks.map((_, i) => ({ page_slug: slug, chunk_index: i })),
          ids: chunks.map((_, i) => `${slug}#${i}`),
        };
      }
      return { documents: [], metadatas: [], ids: [] };
    }),
  } as unknown as Collection;
  return mock;
}

describe("ChromaFsBackend.create", () => {
  it("bootstraps successfully", async () => {
    const backend = await ChromaFsBackend.create(makeCollection());
    expect(backend).toBeInstanceOf(ChromaFsBackend);
  });
});

describe("ChromaFsBackend.ls", () => {
  let backend: ChromaFsBackend;

  beforeEach(async () => {
    backend = await ChromaFsBackend.create(makeCollection());
  });

  it("lists root directory", async () => {
    const result = await backend.ls("/");
    expect(result.error).toBeUndefined();
    expect(result.files?.map((f) => f.path)).toContain("/auth");
    expect(result.files?.map((f) => f.path)).toContain("/public");
  });

  it("lists subdirectory", async () => {
    const result = await backend.ls("/auth");
    expect(result.error).toBeUndefined();
    expect(result.files?.map((f) => f.path)).toContain("/auth/oauth.md");
    expect(result.files?.map((f) => f.path)).toContain("/auth/api-keys.mdx");
  });

  it("returns error for non-existent path", async () => {
    const result = await backend.ls("/nonexistent");
    expect(result.error).toBeDefined();
  });

  it("returns error when path is a file", async () => {
    const result = await backend.ls("/auth/oauth.md");
    expect(result.error).toMatch(/not a directory/i);
  });
});

describe("ChromaFsBackend.read", () => {
  let backend: ChromaFsBackend;

  beforeEach(async () => {
    backend = await ChromaFsBackend.create(makeCollection());
  });

  it("reads file content", async () => {
    const result = await backend.read("/auth/oauth.md");
    expect(result.error).toBeUndefined();
    expect(result.content).toContain("OAuth2");
  });

  it("returns error for non-existent file", async () => {
    const result = await backend.read("/nonexistent.md");
    expect(result.error).toMatch(/no such file/i);
  });

  it("returns error when path is a directory", async () => {
    const result = await backend.read("/auth");
    expect(result.error).toMatch(/is a directory/i);
  });

  it("slices content by offset and limit", async () => {
    const result = await backend.read("/auth/oauth.md", 1, 1);
    expect(result.error).toBeUndefined();
    expect(result.content).toBe("This uses OAuth2 token");
  });
});

describe("ChromaFsBackend.glob", () => {
  let backend: ChromaFsBackend;

  beforeEach(async () => {
    backend = await ChromaFsBackend.create(makeCollection());
  });

  it("matches **/*.md pattern", async () => {
    const result = await backend.glob("**/*.md");
    const paths = result.files?.map((f) => f.path) ?? [];
    expect(paths).toContain("/auth/oauth.md");
    expect(paths).toContain("/public/index.md");
    expect(paths).not.toContain("/auth/api-keys.mdx");
  });

  it("matches **/*.{md,mdx}", async () => {
    const result = await backend.glob("**/*.{md,mdx}");
    const paths = result.files?.map((f) => f.path) ?? [];
    expect(paths).toContain("/auth/api-keys.mdx");
  });
});

describe("ChromaFsBackend — write operations (EROFS)", () => {
  let backend: ChromaFsBackend;

  beforeEach(async () => {
    backend = await ChromaFsBackend.create(makeCollection());
  });

  it("write returns error", async () => {
    const result = await backend.write("/auth/oauth.md", "new content");
    expect(result.error).toMatch(/read-only/i);
  });

  it("edit returns error", async () => {
    const result = await backend.edit("/auth/oauth.md", "old", "new");
    expect(result.error).toMatch(/read-only/i);
  });

  it("uploadFiles returns permission_denied for each file", async () => {
    const result = await backend.uploadFiles([["/auth/oauth.md", new Uint8Array()]]);
    expect(result[0]?.error).toBe("permission_denied");
  });
});

describe("ChromaFsBackend RBAC", () => {
  it("hides private paths without user groups", async () => {
    const backend = await ChromaFsBackend.create(makeCollection());
    const result = await backend.ls("/");
    const paths = result.files?.map((f) => f.path) ?? [];
    expect(paths).not.toContain("/internal");
  });

  it("shows private paths with matching user group", async () => {
    const backend = await ChromaFsBackend.create(makeCollection(), {
      userGroups: new Set(["admin"]),
    });
    const result = await backend.ls("/");
    const paths = result.files?.map((f) => f.path) ?? [];
    expect(paths).toContain("/internal");
  });
});
