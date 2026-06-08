#!/usr/bin/env node
/**
 * Inspect indexed chunks in Milvus and print the LLM-generated context blurb
 * (metadata.context) produced by contextual retrieval.
 *
 * Reuses the project's own Milvus client so it works with both local Milvus
 * (MILVUS_ADDRESS) and Zilliz Cloud (MILVUS_TOKEN auto-resolves the address).
 *
 * Usage:
 *   MILVUS_TOKEN=... node scripts/inspect-milvus.js [codebasePath] [--limit N] [--with-context] [--json]
 *
 *   codebasePath    Path that was indexed (default: current working directory)
 *   --limit N       Max rows to print (default: 20)
 *   --with-context  Only show chunks that actually have an LLM context blurb
 *   --json          Emit raw JSON instead of the formatted view
 *
 * Requires the core package to be built (packages/core/dist). Run `pnpm build:core` first if needed.
 */
const path = require('path');
const corePath = path.resolve(__dirname, '../packages/core/dist/index.js');
let MilvusVectorDatabase, Context;
try {
    ({ MilvusVectorDatabase, Context } = require(corePath));
} catch (err) {
    console.error(`❌ Could not load core from ${corePath}. Build it first: pnpm build:core`);
    console.error(`   (${err instanceof Error ? err.message : err})`);
    process.exit(1);
}

function parseArgs(argv) {
    const opts = { codebasePath: process.cwd(), limit: 20, withContext: false, json: false, id: undefined };
    const positional = [];
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--limit') opts.limit = parseInt(argv[++i], 10) || opts.limit;
        else if (a === '--with-context') opts.withContext = true;
        else if (a === '--json') opts.json = true;
        else if (a === '--id') opts.id = argv[++i];
        else positional.push(a);
    }
    if (positional[0]) opts.codebasePath = path.resolve(positional[0]);
    return opts;
}

async function main() {
    const opts = parseArgs(process.argv.slice(2));

    const address = process.env.MILVUS_ADDRESS;
    const token = process.env.MILVUS_TOKEN;
    if (!address && !token) {
        console.error('❌ Set MILVUS_ADDRESS (local) or MILVUS_TOKEN (Zilliz Cloud) in the environment.');
        process.exit(1);
    }

    const db = new MilvusVectorDatabase({ address, token });

    // Reuse the project's collection-naming (honors HYBRID_MODE + path hash +
    // CODE_CHUNKS_COLLECTION_NAME_OVERRIDE) so we hit the same collection the
    // MCP server indexed into.
    const context = new Context({ vectorDatabase: db });
    const collectionName = context.getCollectionName(opts.codebasePath);

    const exists = await db.hasCollection(collectionName);
    if (!exists) {
        console.error(`❌ Collection '${collectionName}' not found for path '${opts.codebasePath}'.`);
        console.error('   Check the path matches what you indexed, and HYBRID_MODE / CODE_CHUNKS_COLLECTION_NAME_OVERRIDE match.');
        process.exit(1);
    }

    console.error(`🔎 Collection: ${collectionName}`);
    // Look up a single chunk by primary key, or list the first N rows.
    const filter = opts.id ? `id == "${opts.id.replace(/"/g, '')}"` : '';
    const rows = await db.query(
        collectionName,
        filter,
        ['id', 'relativePath', 'startLine', 'endLine', 'content', 'metadata'],
        opts.id ? 1 : opts.limit
    );

    if (opts.id && rows.length === 0) {
        console.error(`❌ No chunk with id '${opts.id}' in '${collectionName}'.`);
        process.exit(1);
    }

    // When fetching one chunk by id, show the full content untruncated.
    const full = Boolean(opts.id);
    const parsed = rows.map((r) => {
        let meta = {};
        try { meta = JSON.parse(r.metadata || '{}'); } catch { /* leave empty */ }
        return {
            id: r.id,
            relativePath: r.relativePath,
            startLine: r.startLine,
            endLine: r.endLine,
            context: meta.context || null,
            content: full ? (r.content || '') : (r.content || '').slice(0, 200),
        };
    });

    const shown = opts.withContext ? parsed.filter((p) => p.context) : parsed;

    if (opts.json) {
        console.log(JSON.stringify(shown, null, 2));
        return;
    }

    const withCtx = parsed.filter((p) => p.context).length;
    console.error(`📊 ${parsed.length} rows fetched, ${withCtx} have an LLM context blurb.\n`);

    for (const p of shown) {
        console.log(`── ${p.id}  ${p.relativePath}:${p.startLine}-${p.endLine}`);
        console.log(`   🧩 context: ${p.context ? p.context : '(none — indexed without contextual retrieval)'}`);
        if (full) {
            console.log(`   📄 content:\n${p.content}\n`);
        } else {
            console.log(`   📄 code: ${p.content.replace(/\n/g, ' ')}…\n`);
        }
    }
}

main().catch((err) => {
    console.error('❌ Failed:', err instanceof Error ? err.message : err);
    process.exit(1);
});
