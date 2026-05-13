/*
ChromaFs Quickstart (JavaScript)

Requires:
  npm install chromadb deepagents-chromafs openai

Run ChromaDB (shared ./examples/chroma-data):
  docker run -d -v ./examples/chroma-data:/data -p 8000:8000 chromadb/chroma

Then:
  OPENAI_API_KEY=... node examples/typescript/quickstar.js
*/

async function main() {
  const { ChromaClient } = await import("chromadb");
  const { ChromaFsBackend } = await import("deepagents-chromafs");
  const OpenAI = (await import("openai")).default;

  const HOST = process.env.CHROMA_HOST ?? "localhost";
  const PORT = Number(process.env.CHROMA_PORT ?? "8000");
  const COLLECTION_NAME = "docs";

  const OPENAI_API_KEY = process.env.OPENAI_API_KEY ?? "";
  const EMBEDDING_MODEL = "text-embedding-3-large";
  const EMBEDDING_DIM = 3072;

  if (!OPENAI_API_KEY) {
    throw new Error("Missing OPENAI_API_KEY. Export it before running this script.");
  }

  const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

  async function embed(texts) {
    const response = await openai.embeddings.create({ model: EMBEDDING_MODEL, input: texts });
    return response.data.map((item) => item.embedding);
  }

  const client = new ChromaClient({ path: `http://${HOST}:${PORT}` });
  const collection = await client.getOrCreateCollection({
    name: COLLECTION_NAME,
    embeddingFunction: null,
  });

  console.log(`Collection '${collection.name}' ready — ${await collection.count()} documents`);

  const pathTree = {
    "data/paris.md": { isPublic: true, groups: [] },
    "data/vietnam.md": { isPublic: true, groups: [] },
  };

  const chunksParis = [
    "Paris, the capital of France, is globally renowned as the 'City of Light' — a city where art, history, and everyday life blend seamlessly.",
    "At its heart stands the Eiffel Tower, the most iconic architectural landmark in the city, drawing millions of visitors from around the world each year.",
    "Beyond the monuments, Parisian cafe culture and freshly baked croissants define the rhythm of daily life for the city's residents.",
    "A short walk from any arrondissement leads to the Louvre Museum, home to priceless masterpieces including the legendary Mona Lisa.",
    "The Seine River ties it all together, winding through the city and framing its historic bridges in a landscape that has inspired artists and lovers for centuries.",
  ];

  const chunksVietnam = [
    "Vietnam is a Southeast Asian country of remarkable contrasts — its long coastline, lush mountains, and river deltas together form one of the most diverse landscapes in the region.",
    "This natural richness is mirrored in Vietnamese cuisine, celebrated worldwide for its fresh ingredients and balance of flavors, with Pho standing as the country's most iconic dish.",
    "Nowhere is Vietnam's natural grandeur more dramatic than Ha Long Bay, a UNESCO World Heritage site where thousands of limestone islands rise from emerald waters.",
    "The people who call this land home have long been admired for their resilience and the genuine warmth with which they welcome strangers.",
    "That spirit is most alive in Hanoi and Ho Chi Minh City, two great metropolises where ancient temple districts sit side by side with gleaming modern towers.",
  ];

  await collection.upsert({
    ids: ["__path_tree__"],
    documents: [JSON.stringify(pathTree)],
    embeddings: [Array(EMBEDDING_DIM).fill(0)],
    metadatas: [{ _system: true }],
  });
  console.log("Path tree inserted");

  async function buildChunkRecords(slug, chunks) {
    const ids = chunks.map((_, index) => `${slug}::${index}`);
    const metadatas = chunks.map((_, index) => ({ page_slug: slug, chunk_index: index }));
    const embeddings = await embed(chunks);
    return { ids, documents: chunks, metadatas, embeddings };
  }

  for (const [slug, chunks] of [
    ["data/paris.md", chunksParis],
    ["data/vietnam.md", chunksVietnam],
  ]) {
    const records = await buildChunkRecords(slug, chunks);
    await collection.upsert(records);
    console.log(`Inserted ${records.ids.length} chunks for ${slug}`);
  }

  const backend = await ChromaFsBackend.create(collection);

  const root = await backend.ls("/");
  const data = await backend.ls("/data");
  console.log("ls /:", root.files);
  console.log("ls /data:", data.files);

  const readResult = await backend.read("/data/paris.md");
  if (readResult.error) {
    throw new Error(readResult.error);
  }
  console.log("\n--- /data/paris.md ---");
  console.log(readResult.content);

  const grepResult = await backend.grep("UNESCO");
  console.log("\ngrep 'UNESCO':");
  for (const match of grepResult.matches ?? []) {
    console.log(`  ${match.path}:${match.line}: ${match.text}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
