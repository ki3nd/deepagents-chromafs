/**
 * Grep pipeline for ChromaFs (4-step algorithm).
 *
 * 1. toChromaFilter  — build Chroma where_document filter
 * 2. findMatchingSlugs — coarse filter via Chroma $contains / $regex
 * 3. bulkPrefetch    — concurrent fetch of matched pages into ContentCache
 * 4. fineFilter      — in-memory regex for line-level GrepMatch results
 */

import { IncludeEnum } from "chromadb";
import type { Collection, Where, WhereDocument } from "chromadb";
import type { GrepMatch } from "deepagents";
import type { ContentCache } from "./cache.js";

const SLUG_BATCH_SIZE = 500;

// ---------------------------------------------------------------------------
// Step 1 — build the Chroma where_document filter
// ---------------------------------------------------------------------------

export type ChromaFilter = WhereDocument;

function splitLines(content: string): string[] {
  if (content.length === 0) return [];
  const lines = content.split(/\r\n|\r|\n/);
  if (lines.length > 0 && lines[lines.length - 1] === "") {
    lines.pop();
  }
  return lines;
}

export function toChromaFilter(
  pattern: string,
  options: { fixedString?: boolean; ignoreCase?: boolean } = {},
): ChromaFilter {
  const { fixedString = true, ignoreCase = false } = options;

  if (fixedString && !ignoreCase) {
    return { $contains: pattern };
  }

  if (fixedString && ignoreCase) {
    return { $regex: `(?i)${escapeRegex(pattern)}` } as ChromaFilter;
  }

  if (ignoreCase) {
    return { $regex: `(?i)${pattern}` } as ChromaFilter;
  }

  return { $regex: pattern } as ChromaFilter;
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// ---------------------------------------------------------------------------
// Step 2 — coarse filter: find slugs whose chunks match the pattern
// ---------------------------------------------------------------------------

async function queryChromaBatch(
  collection: Collection,
  chromaFilter: ChromaFilter,
  slugBatch: string[],
  slugField: string,
): Promise<string[]> {
  const where: Where = { [slugField]: { $in: slugBatch } };
  const results = await collection.get({
    where,
    whereDocument: chromaFilter,
    include: [IncludeEnum.Metadatas],
  });
  const metadatas = results.metadatas ?? [];
  return metadatas
    .filter((m): m is NonNullable<typeof m> => m !== null && slugField in m)
    .map((m) => String(m[slugField]));
}

export async function findMatchingSlugs(
  collection: Collection,
  chromaFilter: ChromaFilter,
  candidateSlugs: string[],
  options: { slugField?: string } = {},
): Promise<string[]> {
  const { slugField = "page_slug" } = options;

  if (candidateSlugs.length === 0) {
    const results = await collection.get({
      whereDocument: chromaFilter,
      include: [IncludeEnum.Metadatas],
    });
    const metadatas = results.metadatas ?? [];
    const matched = new Set(
      metadatas
        .filter((m): m is NonNullable<typeof m> => m !== null && slugField in m)
        .map((m) => String(m[slugField])),
    );
    return [...matched].sort();
  }

  const matched = new Set<string>();
  for (let i = 0; i < candidateSlugs.length; i += SLUG_BATCH_SIZE) {
    const batch = candidateSlugs.slice(i, i + SLUG_BATCH_SIZE);
    const slugs = await queryChromaBatch(collection, chromaFilter, batch, slugField);
    for (const s of slugs) matched.add(s);
  }
  return [...matched].sort();
}

// ---------------------------------------------------------------------------
// Step 3 — bulk prefetch: reassemble pages into the cache
// ---------------------------------------------------------------------------

export async function fetchPage(
  collection: Collection,
  slug: string,
  options: { slugField?: string; chunkIndexField?: string } = {},
): Promise<string> {
  const { slugField = "page_slug", chunkIndexField = "chunk_index" } = options;

  const where: Where = { [slugField]: slug };
  const results = await collection.get({
    where,
    include: [IncludeEnum.Documents, IncludeEnum.Metadatas],
  });

  const documents = results.documents ?? [];
  const metadatas = results.metadatas ?? [];

  if (documents.length === 0) return "";

  const indexed: Array<[number, string]> = [];
  const pairCount = Math.min(documents.length, metadatas.length);
  for (let i = 0; i < pairCount; i++) {
    const doc = documents[i];
    const meta = metadatas[i];
    if (doc == null) continue;
    const idx = meta != null && chunkIndexField in meta ? Number(meta[chunkIndexField]) : 0;
    indexed.push([idx, doc]);
  }

  indexed.sort((a, b) => a[0] - b[0]);
  return indexed.map(([, text]) => text).join("\n");
}

export async function bulkPrefetch(
  collection: Collection,
  slugs: string[],
  cache: ContentCache,
  options: { slugField?: string; chunkIndexField?: string } = {},
): Promise<void> {
  for (const slug of slugs) {
    if (cache.has(slug)) continue;
    const content = await fetchPage(collection, slug, options);
    cache.put(slug, content);
  }
}

// ---------------------------------------------------------------------------
// Step 4 — fine filter: in-memory regex on cached content
// ---------------------------------------------------------------------------

export function fineFilter(
  pattern: string,
  slugs: string[],
  cache: ContentCache,
  options: {
    fixedString?: boolean;
    ignoreCase?: boolean;
    slugToPath?: (slug: string) => string;
  } = {},
): GrepMatch[] {
  const { fixedString = true, ignoreCase = false, slugToPath } = options;

  const flags = ignoreCase ? "i" : "";
  const effectivePattern = fixedString ? escapeRegex(pattern) : pattern;

  let regex: RegExp;
  try {
    regex = new RegExp(effectivePattern, flags);
  } catch {
    return [];
  }

  const resolvePath = slugToPath ?? ((slug: string) => "/" + slug);
  const matches: GrepMatch[] = [];

  for (const slug of slugs) {
    const content = cache.get(slug);
    if (content == null) continue;
    const path = resolvePath(slug);
    const lines = splitLines(content);
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i] ?? "";
      if (regex.test(line)) {
        matches.push({ path, line: i + 1, text: line });
      }
    }
  }

  return matches;
}

export function parseGroupsMetadata(raw: string | string[] | null | undefined): string[] {
  if (raw == null) return [];
  if (Array.isArray(raw)) return raw.map(String);

  try {
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed) ? parsed.map(String) : [];
  } catch {
    return [];
  }
}
