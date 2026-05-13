/**
 * ChromaFsBackend: read-only BackendProtocolV2 backed by a ChromaDB collection.
 *
 * Implements the ChromaFs algorithm from Mintlify: the entire directory
 * tree is stored as a single compressed JSON document in Chroma (__path_tree__),
 * allowing instant bootstrap (~100 ms) without spinning up a sandbox.
 *
 * All write-related methods (write, edit, uploadFiles) return EROFS error.
 */

import { IncludeEnum } from "chromadb";
import type { Collection } from "chromadb";
import type {
  BackendProtocolV2,
  EditResult,
  FileDownloadResponse,
  FileInfo,
  FileUploadResponse,
  GlobResult,
  GrepResult,
  LsResult,
  ReadRawResult,
  ReadResult,
  WriteResult,
} from "deepagents";

import { ContentCache } from "./cache.js";
import { bulkPrefetch, fetchPage, findMatchingSlugs, fineFilter, toChromaFilter } from "./grep.js";
import { PathTree } from "./tree.js";

const PATH_TREE_ID = "__path_tree__";
const EROFS_ERROR = "Read-only filesystem: ChromaFsBackend does not support write operations.";

function splitLines(content: string): string[] {
  if (content.length === 0) return [];
  const lines = content.split(/\r\n|\r|\n/);
  if (lines.length > 0 && lines[lines.length - 1] === "") {
    lines.pop();
  }
  return lines;
}

export interface ChromaFsBackendOptions {
  userGroups?: Set<string>;
  slugField?: string;
  chunkIndexField?: string;
  cache?: ContentCache;
}

interface AsyncContentCacheLike {
  getAsync?: (slug: string) => Promise<string | undefined>;
  putAsync?: (slug: string, content: string) => Promise<void>;
  hasAsync?: (slug: string) => Promise<boolean>;
}

export class ChromaFsBackend implements BackendProtocolV2 {
  private readonly collection: Collection;
  private readonly slugField: string;
  private readonly chunkIndexField: string;
  private readonly cache: ContentCache;
  private tree: PathTree;

  private constructor(
    collection: Collection,
    options: ChromaFsBackendOptions,
    tree: PathTree,
  ) {
    this.collection = collection;
    this.slugField = options.slugField ?? "page_slug";
    this.chunkIndexField = options.chunkIndexField ?? "chunk_index";
    this.cache = options.cache ?? new ContentCache();
    this.tree = tree;
  }

  /**
   * Create and bootstrap a ChromaFsBackend instance.
   * Fetches __path_tree__ from Chroma and builds the in-memory directory index.
   */
  static async create(
    collection: Collection,
    options: ChromaFsBackendOptions = {},
  ): Promise<ChromaFsBackend> {
    const tree = await ChromaFsBackend._bootstrapTree(collection, options.userGroups);
    return new ChromaFsBackend(collection, options, tree);
  }

  private static async _bootstrapTree(
    collection: Collection,
    userGroups?: Set<string>,
  ): Promise<PathTree> {
    const results = await collection.get({
      ids: [PATH_TREE_ID],
      include: [IncludeEnum.Documents],
    });
    const documents = results.documents ?? [];
    const first = documents[0];
    if (!first) return PathTree.empty();
    return PathTree.fromJson(first, userGroups);
  }

  private async fetchAndCache(slug: string): Promise<string> {
    const asyncCache = this.cache as ContentCache & AsyncContentCacheLike;

    if (typeof asyncCache.hasAsync === "function") {
      const hasEntry = await asyncCache.hasAsync(slug);
      if (!hasEntry) {
        const content = await fetchPage(this.collection, slug, {
          slugField: this.slugField,
          chunkIndexField: this.chunkIndexField,
        });
        if (typeof asyncCache.putAsync === "function") {
          await asyncCache.putAsync(slug, content);
        } else {
          this.cache.put(slug, content);
        }
      }

      if (typeof asyncCache.getAsync === "function") {
        return (await asyncCache.getAsync(slug)) ?? "";
      }
      return this.cache.get(slug) ?? "";
    }

    if (!this.cache.has(slug)) {
      const content = await fetchPage(this.collection, slug, {
        slugField: this.slugField,
        chunkIndexField: this.chunkIndexField,
      });
      this.cache.put(slug, content);
    }
    return this.cache.get(slug) ?? "";
  }

  // --------------------------------------------------------------------------
  // BackendProtocolV2 — read operations
  // --------------------------------------------------------------------------

  async ls(path: string): Promise<LsResult> {
    const normalised = path.replace(/\/+$/, "") || "/";
    if (!this.tree.isDir(normalised)) {
      if (this.tree.isFile(normalised)) {
        return { error: `Not a directory: ${path}` };
      }
      return { error: `No such directory: ${path}` };
    }

    const files: FileInfo[] = this.tree.lsChildren(normalised).map((child) => ({
      path: child,
      is_dir: this.tree.isDir(child),
    }));
    return { files };
  }

  async readRaw(filePath: string): Promise<ReadRawResult> {
    const normalised = filePath.replace(/\/+$/, "");
    if (!this.tree.isFile(normalised)) {
      if (this.tree.isDir(normalised)) return { error: `Is a directory: ${filePath}` };
      return { error: `No such file: ${filePath}` };
    }
    const slug = this.tree.pathToSlug(normalised);
    const content = await this.fetchAndCache(slug);
    const now = new Date().toISOString();
    return {
      data: {
        content: splitLines(content),
        created_at: now,
        modified_at: now,
      },
    };
  }

  async read(filePath: string, offset = 0, limit = 2000): Promise<ReadResult> {
    const normalised = filePath.replace(/\/+$/, "");
    if (!this.tree.isFile(normalised)) {
      if (this.tree.isDir(normalised)) {
        return { error: `Is a directory: ${filePath}` };
      }
      return { error: `No such file: ${filePath}` };
    }

    const slug = this.tree.pathToSlug(normalised);
    const content = await this.fetchAndCache(slug);
    const lines = splitLines(content);

    if (offset > 0 && lines.length > 0 && offset >= lines.length) {
      return { error: `Line offset ${offset} exceeds file length (${lines.length} lines)` };
    }

    const sliced = lines.slice(offset, offset + limit).join("\n");
    return { content: sliced, mimeType: "text/plain" };
  }

  async grep(
    pattern: string,
    path?: string | null,
    glob?: string | null,
  ): Promise<GrepResult> {
    const searchRoot = (path ?? "/").replace(/\/+$/, "") || "/";
    const candidatePaths = glob
      ? this.tree.globMatch(searchRoot, glob)
      : this.tree.slugsUnder(searchRoot);
    const candidateSlugs = candidatePaths.map((p) => this.tree.pathToSlug(p));

    const chromaFilter = toChromaFilter(pattern, { fixedString: true });
    const matchedSlugs = await findMatchingSlugs(this.collection, chromaFilter, candidateSlugs, {
      slugField: this.slugField,
    });

    if (matchedSlugs.length === 0) return { matches: [] };

    const asyncCache = this.cache as ContentCache & AsyncContentCacheLike;
    if (typeof asyncCache.hasAsync === "function") {
      const warmedCache = new ContentCache();
      for (const slug of matchedSlugs) {
        const content = await this.fetchAndCache(slug);
        warmedCache.put(slug, content);
      }

      const matches = fineFilter(pattern, matchedSlugs, warmedCache, {
        fixedString: true,
        slugToPath: (s) => this.tree.slugToPath(s),
      });
      return { matches };
    }

    await bulkPrefetch(this.collection, matchedSlugs, this.cache, {
      slugField: this.slugField,
      chunkIndexField: this.chunkIndexField,
    });

    const matches = fineFilter(pattern, matchedSlugs, this.cache, {
      fixedString: true,
      slugToPath: (s) => this.tree.slugToPath(s),
    });
    return { matches };
  }

  async glob(pattern: string, path = "/"): Promise<GlobResult> {
    const matched = this.tree.globMatch(path, pattern);
    const files: FileInfo[] = matched.map((p) => ({ path: p, is_dir: false }));
    return { files };
  }

  async downloadFiles(paths: string[]): Promise<FileDownloadResponse[]> {
    return Promise.all(
      paths.map(async (path): Promise<FileDownloadResponse> => {
        const normalised = path.replace(/\/+$/, "");
        if (this.tree.isDir(normalised)) {
          return { path, content: null, error: "is_directory" };
        }
        if (!this.tree.isFile(normalised)) {
          return { path, content: null, error: "file_not_found" };
        }
        const slug = this.tree.pathToSlug(normalised);
        const content = await this.fetchAndCache(slug);
        return { path, content: Buffer.from(content, "utf-8"), error: null };
      }),
    );
  }

  // --------------------------------------------------------------------------
  // BackendProtocolV2 — write operations (EROFS)
  // --------------------------------------------------------------------------

  async write(_filePath: string, _content: string): Promise<WriteResult> {
    return { error: EROFS_ERROR };
  }

  async edit(
    _filePath: string,
    _oldString: string,
    _newString: string,
    _replaceAll?: boolean,
  ): Promise<EditResult> {
    return { error: EROFS_ERROR };
  }

  async uploadFiles(files: Array<[string, Uint8Array]>): Promise<FileUploadResponse[]> {
    return files.map(([path]) => ({ path, error: "permission_denied" as const }));
  }
}
