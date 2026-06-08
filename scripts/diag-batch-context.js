#!/usr/bin/env node
/**
 * Live diagnostic for the Gemini file-based batch contextualizer.
 *
 * Exercises the exact path used during indexing (contextualizeBatchViaBatchApi)
 * on two trivial chunks, with verbose heartbeat logging, and prints the result.
 * The indexer swallows batch failures (so a broken batch path just produces "no
 * context"); this script surfaces the real error instead.
 *
 * Usage:
 *   GEMINI_API_KEY=... node scripts/diag-batch-context.js
 *   (optional) CONTEXTUAL_RETRIEVAL_MODEL=gemini-2.5-flash-lite GEMINI_BASE_URL=...
 *
 * Requires the core package to be built (pnpm build:core).
 */
const path = require('path');
const { GeminiContextualizer } = require(path.resolve(__dirname, '../packages/core/dist/index.js'));

(async () => {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
        console.error('❌ Set GEMINI_API_KEY in the environment.');
        process.exit(1);
    }
    const model = process.env.CONTEXTUAL_RETRIEVAL_MODEL || 'gemini-2.5-flash-lite';
    const baseURL = process.env.GEMINI_BASE_URL;

    const ctx = new GeminiContextualizer({
        apiKey,
        model,
        ...(baseURL ? { baseURL } : {}),
        batchPollIntervalMs: 5000,
        batchMaxWaitMs: 10 * 60 * 1000, // 10 min cap for the diagnostic
    });

    const file = 'export function add(a, b) { return a + b; }\nexport function sub(a, b) { return a - b; }\n';
    const items = [
        { chunkContent: 'export function add(a, b) { return a + b; }', fileContent: file, relativePath: 'math.ts' },
        { chunkContent: 'export function sub(a, b) { return a - b; }', fileContent: file, relativePath: 'math.ts' },
    ];

    console.log(`🧪 Submitting 2 chunks via file-based Gemini Batch API (model: ${model})...`);
    console.log('   (batch jobs are async; this can take a few minutes for tiny inputs)\n');

    const t0 = Date.now();
    const res = await ctx.contextualizeBatchViaBatchApi(items, (p) => {
        console.log(`   ⏳ heartbeat: job ${p.jobNum}/${p.jobTotal} ${p.state} (${Math.round(p.elapsedMs / 1000)}s)`);
    });

    console.log(`\n⏱  finished in ${Math.round((Date.now() - t0) / 1000)}s`);
    console.log('RESULT:', JSON.stringify(res, null, 2));

    if (res.every((r) => r)) {
        console.log('\n✅ File-based batch path WORKS. If indexing still shows no context, the issue is server-side: env not set (CONTEXTUAL_RETRIEVAL_ENABLED / CONTEXTUAL_RETRIEVAL_BATCH / GEMINI_API_KEY) or a stale MCP server running an older build.');
    } else {
        console.log('\n❌ Batch path returned EMPTY context. The real error is in the warning(s) above (now including the job error detail). That is why indexing wrote no metadata.context.');
    }
})().catch((e) => {
    console.error('THREW:', e);
    process.exit(1);
});
