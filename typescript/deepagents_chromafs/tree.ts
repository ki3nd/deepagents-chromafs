/**
 * PathTree: in-memory virtual filesystem tree built from ChromaDB path metadata.
 *
 * Bootstrapped once from the `__path_tree__` document stored in the Chroma
 * collection, then used for all `ls`, `glob`, and directory-scope operations
 * with zero network calls.
 *
 * Path tree document format:
 *   {
 *     "auth/oauth.md": { "isPublic": true, "groups": [] },
 *     "internal/billing.md": { "isPublic": false, "groups": ["admin"] }
 *   }
 *
 * The document may optionally be gzip-compressed and base64-encoded.
 */

import { gunzipSync } from "node:zlib";
import micromatch from "micromatch";

export interface PathInfoData {
  isPublic?: boolean;
  groups?: string[];
}

export interface PathInfo {
  isPublic: boolean;
  groups: Set<string>;
}

function parsePathInfo(data: PathInfoData): PathInfo {
  const rawGroups = Array.isArray(data.groups) ? data.groups : [];
  return {
    isPublic: data.isPublic !== false,
    groups: new Set(rawGroups.map(String)),
  };
}

function isAccessible(info: PathInfo, userGroups: Set<string>): boolean {
  if (info.isPublic) return true;
  for (const g of info.groups) {
    if (userGroups.has(g)) return true;
  }
  return false;
}

function decompressTreeDocument(raw: string): string {
  const stripped = raw.trim();
  if (stripped.startsWith("{")) return stripped;
  try {
    const compressed = Buffer.from(stripped, "base64");
    return gunzipSync(compressed).toString("utf-8");
  } catch (err) {
    throw new Error(`Cannot decode path tree document: ${String(err)}`);
  }
}

export class PathTree {
  private readonly filePaths: Set<string>;
  private readonly dirChildren: Map<string, string[]>;

  private constructor(filePaths: Set<string>, dirChildren: Map<string, string[]>) {
    this.filePaths = filePaths;
    this.dirChildren = dirChildren;
  }

  static empty(): PathTree {
    return new PathTree(new Set(), new Map());
  }

  static fromJson(raw: string, userGroups?: Set<string>): PathTree {
    const jsonStr = decompressTreeDocument(raw);
    const data = JSON.parse(jsonStr) as Record<string, PathInfoData>;
    const effectiveGroups = userGroups ?? new Set<string>();

    const filePaths = new Set<string>();
    for (const [slug, infoData] of Object.entries(data)) {
      const info = parsePathInfo(infoData);
      if (isAccessible(info, effectiveGroups)) {
        const cleanSlug = slug.replace(/^\/+/, "");
        filePaths.add("/" + cleanSlug);
      }
    }

    return PathTree._build(filePaths);
  }

  private static _build(filePaths: Set<string>): PathTree {
    const dirChildren = new Map<string, string[]>();

    for (const filePath of filePaths) {
      const parts = filePath.split("/").filter(Boolean);
      // parts = ["auth", "oauth.md"] for "/auth/oauth.md"
      for (let depth = 0; depth < parts.length; depth++) {
        const parent = depth === 0 ? "/" : "/" + parts.slice(0, depth).join("/");
        const child = "/" + parts.slice(0, depth + 1).join("/");
        if (!dirChildren.has(parent)) dirChildren.set(parent, []);
        const children = dirChildren.get(parent)!;
        if (!children.includes(child)) children.push(child);
      }
    }

    // Sort children lists for deterministic output
    for (const children of dirChildren.values()) {
      children.sort();
    }

    return new PathTree(filePaths, dirChildren);
  }

  isFile(path: string): boolean {
    return this.filePaths.has(path.replace(/\/+$/, ""));
  }

  isDir(path: string): boolean {
    const normalised = path.replace(/\/+$/, "") || "/";
    return this.dirChildren.has(normalised);
  }

  lsChildren(path: string): string[] {
    const normalised = path.replace(/\/+$/, "") || "/";
    return [...(this.dirChildren.get(normalised) ?? [])];
  }

  slugsUnder(path: string): string[] {
    const normalised = (path.replace(/\/+$/, "") || "/") + "/";
    const prefix = normalised === "//" ? "/" : normalised;

    const results: string[] = [];
    for (const fp of this.filePaths) {
      if (prefix === "/" || fp.startsWith(prefix)) {
        results.push(fp);
      }
    }
    return results.sort();
  }

  globMatch(path: string, pattern: string): string[] {
    const base = path.replace(/\/+$/, "") || "/";
    const candidates = this.slugsUnder(base);
    const basePrefix = base === "/" ? "" : base;

    const matches: string[] = [];
    for (const fp of candidates) {
      const relative = fp.slice(basePrefix.length).replace(/^\//, "");
      if (micromatch.isMatch(relative, pattern)) {
        matches.push(fp);
      }
    }
    return matches;
  }

  pathToSlug(path: string): string {
    return path.replace(/^\/+/, "");
  }

  slugToPath(slug: string): string {
    return "/" + slug.replace(/^\/+/, "");
  }
}
