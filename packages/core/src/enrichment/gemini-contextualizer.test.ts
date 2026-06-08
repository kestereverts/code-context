import * as fs from 'fs';
import { GoogleGenAI } from '@google/genai';
import { GeminiContextualizer } from './gemini-contextualizer';

const mockGenerateContent = jest.fn();
const mockBatchCreate = jest.fn();
const mockBatchGet = jest.fn();
const mockBatchCancel = jest.fn();
const mockFilesUpload = jest.fn();
const mockFilesDownload = jest.fn();

jest.mock('@google/genai', () => ({
    GoogleGenAI: jest.fn().mockImplementation(() => ({
        models: {
            generateContent: mockGenerateContent
        },
        batches: {
            create: mockBatchCreate,
            get: mockBatchGet,
            cancel: mockBatchCancel
        },
        files: {
            upload: mockFilesUpload,
            download: mockFilesDownload
        }
    }))
}));

describe('GeminiContextualizer', () => {
    beforeEach(() => {
        mockGenerateContent.mockReset();
        mockBatchCreate.mockReset();
        mockBatchGet.mockReset();
        mockBatchCancel.mockReset();
        mockFilesUpload.mockReset();
        mockFilesDownload.mockReset();
        (GoogleGenAI as unknown as jest.Mock).mockClear();
    });

    it('defaults to gemini-2.5-flash-lite and passes the model through', async () => {
        mockGenerateContent.mockResolvedValue({ text: 'Situating context.' });

        const ctx = new GeminiContextualizer({ apiKey: 'test-key' });
        expect(ctx.getModel()).toBe('gemini-2.5-flash-lite');

        const result = await ctx.contextualize({
            chunkContent: 'function add(a, b) { return a + b; }',
            fileContent: 'export function add(a, b) { return a + b; }',
            relativePath: 'src/math.ts'
        });

        expect(result).toBe('Situating context.');
        expect(mockGenerateContent).toHaveBeenCalledTimes(1);
        expect(mockGenerateContent.mock.calls[0][0]).toMatchObject({
            model: 'gemini-2.5-flash-lite'
        });
    });

    it('honors a custom model', () => {
        const ctx = new GeminiContextualizer({ apiKey: 'test-key', model: 'gemini-2.5-flash' });
        expect(ctx.getModel()).toBe('gemini-2.5-flash');
    });

    it('returns empty string on LLM failure so indexing can proceed', async () => {
        mockGenerateContent.mockRejectedValue(new Error('rate limited'));

        const ctx = new GeminiContextualizer({ apiKey: 'test-key' });
        const result = await ctx.contextualize({
            chunkContent: 'chunk',
            fileContent: 'file',
            relativePath: 'a.ts'
        });

        expect(result).toBe('');
    });

    it('preserves input order in batched contextualization', async () => {
        mockGenerateContent.mockImplementation(async (params: any) => {
            // Echo back the chunk content embedded in the prompt to verify ordering.
            const match = /<chunk>\n([\s\S]*?)\n<\/chunk>/.exec(params.contents);
            return { text: `ctx:${match ? match[1] : '?'}` };
        });

        const ctx = new GeminiContextualizer({ apiKey: 'test-key', maxConcurrency: 2 });
        const items = ['one', 'two', 'three', 'four'].map(c => ({
            chunkContent: c,
            fileContent: `file with ${c}`,
            relativePath: 'f.ts'
        }));

        const results = await ctx.contextualizeBatch(items);
        expect(results).toEqual(['ctx:one', 'ctx:two', 'ctx:three', 'ctx:four']);
    });

    it('truncates oversized file content before sending to the LLM', async () => {
        mockGenerateContent.mockResolvedValue({ text: 'ok' });

        const ctx = new GeminiContextualizer({ apiKey: 'test-key', maxFileChars: 10 });
        await ctx.contextualize({
            chunkContent: 'chunk',
            fileContent: 'x'.repeat(1000),
            relativePath: 'big.ts'
        });

        const sentPrompt: string = mockGenerateContent.mock.calls[0][0].contents;
        // The 1000-char file must have been truncated to 10 chars inside the prompt.
        expect(sentPrompt).not.toContain('x'.repeat(11));
        expect(sentPrompt).toContain('x'.repeat(10));
    });

    describe('contextualizeBatchViaBatchApi', () => {
        const items = (n: number) =>
            Array.from({ length: n }, (_, i) => ({
                chunkContent: `chunk ${i}`,
                fileContent: `file ${i}`,
                relativePath: `f${i}.ts`,
            }));

        const candidate = (text: string) => ({ candidates: [{ content: { parts: [{ text }] } }] });

        // Drive a single successful file-based job whose result JSONL is `lines`.
        const succeedWith = (lines: string[]) => {
            mockFilesUpload.mockResolvedValue({ name: 'files/in' });
            mockBatchCreate.mockResolvedValue({ name: 'batches/x' });
            mockBatchGet.mockResolvedValue({ state: 'JOB_STATE_SUCCEEDED', dest: { fileName: 'files/out' } });
            mockFilesDownload.mockImplementation(async ({ downloadPath }: { downloadPath: string }) => {
                fs.writeFileSync(downloadPath, lines.join('\n') + '\n');
            });
        };

        it('uploads JSONL, runs one file job, and returns texts in input order', async () => {
            succeedWith([
                JSON.stringify({ key: 'req-0', response: candidate('ctx-0') }),
                JSON.stringify({ key: 'req-1', response: candidate('ctx-1') }),
                JSON.stringify({ key: 'req-2', response: candidate('ctx-2') }),
            ]);

            const ctx = new GeminiContextualizer({ apiKey: 'k', batchPollIntervalMs: 0 });
            const result = await ctx.contextualizeBatchViaBatchApi(items(3));

            expect(result).toEqual(['ctx-0', 'ctx-1', 'ctx-2']);
            expect(mockFilesUpload).toHaveBeenCalledTimes(1);
            expect(mockBatchCreate).toHaveBeenCalledTimes(1);
            // src must be the uploaded file's resource name (string), not inline requests.
            expect(mockBatchCreate.mock.calls[0][0]).toMatchObject({ model: 'gemini-2.5-flash-lite', src: 'files/in' });
        });

        it('matches results BY KEY regardless of result ordering', async () => {
            // Service returns the lines in a different order than submitted.
            succeedWith([
                JSON.stringify({ key: 'req-2', response: candidate('ctx-2') }),
                JSON.stringify({ key: 'req-0', response: candidate('ctx-0') }),
                JSON.stringify({ key: 'req-1', response: candidate('ctx-1') }),
            ]);

            const ctx = new GeminiContextualizer({ apiKey: 'k', batchPollIntervalMs: 0 });
            const result = await ctx.contextualizeBatchViaBatchApi(items(3));

            expect(result).toEqual(['ctx-0', 'ctx-1', 'ctx-2']);
        });

        it('yields NO context for a dropped result instead of misattributing it', async () => {
            // The exact corruption we are fixing: req-1 is missing from results.
            // Position-based mapping would shift req-2's blurb onto req-1.
            succeedWith([
                JSON.stringify({ key: 'req-0', response: candidate('ctx-0') }),
                JSON.stringify({ key: 'req-2', response: candidate('ctx-2') }),
            ]);

            const ctx = new GeminiContextualizer({ apiKey: 'k', batchPollIntervalMs: 0 });
            const result = await ctx.contextualizeBatchViaBatchApi(items(3));

            // req-1 -> '', and req-2 keeps its own blurb. No misattribution.
            expect(result).toEqual(['ctx-0', '', 'ctx-2']);
        });

        it('splits into multiple file jobs to respect the byte cap', async () => {
            // Each job's result file can safely contain all keys; by-key matching
            // ignores keys outside the current group.
            succeedWith([0, 1, 2].map(i => JSON.stringify({ key: `req-${i}`, response: candidate(`ctx-${i}`) })));

            const ctx = new GeminiContextualizer({ apiKey: 'k', batchPollIntervalMs: 0, batchMaxBytes: 1 });
            const result = await ctx.contextualizeBatchViaBatchApi(items(3));

            expect(mockBatchCreate).toHaveBeenCalledTimes(3);
            expect(mockFilesUpload).toHaveBeenCalledTimes(3);
            expect(result).toEqual(['ctx-0', 'ctx-1', 'ctx-2']);
        });

        it('polls until the job reaches a terminal state', async () => {
            mockFilesUpload.mockResolvedValue({ name: 'files/in' });
            mockBatchCreate.mockResolvedValue({ name: 'batches/poll' });
            mockBatchGet
                .mockResolvedValueOnce({ state: 'JOB_STATE_PENDING' })
                .mockResolvedValueOnce({ state: 'JOB_STATE_RUNNING' })
                .mockResolvedValueOnce({ state: 'JOB_STATE_SUCCEEDED', dest: { fileName: 'files/out' } });
            mockFilesDownload.mockImplementation(async ({ downloadPath }: { downloadPath: string }) => {
                fs.writeFileSync(downloadPath, JSON.stringify({ key: 'req-0', response: candidate('done') }) + '\n');
            });

            const ctx = new GeminiContextualizer({ apiKey: 'k', batchPollIntervalMs: 0 });
            const result = await ctx.contextualizeBatchViaBatchApi(items(1));

            expect(mockBatchGet).toHaveBeenCalledTimes(3);
            expect(result).toEqual(['done']);
        });

        it('cancels and yields empty strings when a job times out', async () => {
            mockFilesUpload.mockResolvedValue({ name: 'files/in' });
            mockBatchCreate.mockResolvedValue({ name: 'batches/slow' });
            mockBatchGet.mockResolvedValue({ state: 'JOB_STATE_RUNNING' });

            const ctx = new GeminiContextualizer({ apiKey: 'k', batchPollIntervalMs: 0, batchMaxWaitMs: 0 });
            const result = await ctx.contextualizeBatchViaBatchApi(items(2));

            expect(result).toEqual(['', '']);
            expect(mockBatchCancel).toHaveBeenCalledWith({ name: 'batches/slow' });
        });

        it('yields empty strings for a failed job without throwing', async () => {
            mockFilesUpload.mockResolvedValue({ name: 'files/in' });
            mockBatchCreate.mockResolvedValue({ name: 'batches/fail' });
            mockBatchGet.mockResolvedValue({ state: 'JOB_STATE_FAILED' });

            const ctx = new GeminiContextualizer({ apiKey: 'k', batchPollIntervalMs: 0 });
            const result = await ctx.contextualizeBatchViaBatchApi(items(2));

            expect(result).toEqual(['', '']);
        });

        it('maps per-line errors to empty strings while keeping good results', async () => {
            succeedWith([
                JSON.stringify({ key: 'req-0', response: candidate('ok-0') }),
                JSON.stringify({ key: 'req-1', error: { code: 500, message: 'boom' } }),
                JSON.stringify({ key: 'req-2', response: candidate('ok-2') }),
            ]);

            const ctx = new GeminiContextualizer({ apiKey: 'k', batchPollIntervalMs: 0 });
            const result = await ctx.contextualizeBatchViaBatchApi(items(3));

            expect(result).toEqual(['ok-0', '', 'ok-2']);
        });

        it('emits a heartbeat with state and elapsed time on each poll tick', async () => {
            mockFilesUpload.mockResolvedValue({ name: 'files/in' });
            mockBatchCreate.mockResolvedValue({ name: 'batches/hb' });
            mockBatchGet
                .mockResolvedValueOnce({ state: 'JOB_STATE_PENDING' })
                .mockResolvedValueOnce({ state: 'JOB_STATE_RUNNING' })
                .mockResolvedValueOnce({ state: 'JOB_STATE_SUCCEEDED', dest: { fileName: 'files/out' } });
            mockFilesDownload.mockImplementation(async ({ downloadPath }: { downloadPath: string }) => {
                fs.writeFileSync(downloadPath, JSON.stringify({ key: 'req-0', response: candidate('done') }) + '\n');
            });

            const ctx = new GeminiContextualizer({ apiKey: 'k', batchPollIntervalMs: 0 });
            const beats: Array<{ state: string; jobNum: number; jobTotal: number; elapsedMs: number }> = [];

            await ctx.contextualizeBatchViaBatchApi(items(1), (p) => beats.push(p));

            expect(beats.map(b => b.state)).toEqual(['JOB_STATE_PENDING', 'JOB_STATE_RUNNING']);
            expect(beats.every(b => b.jobNum === 1 && b.jobTotal === 1)).toBe(true);
            expect(beats.every(b => typeof b.elapsedMs === 'number')).toBe(true);
        });

        it('supports the plain `text` field form in result lines', async () => {
            succeedWith([JSON.stringify({ key: 'req-0', response: { text: 'plain' } })]);

            const ctx = new GeminiContextualizer({ apiKey: 'k', batchPollIntervalMs: 0 });
            const result = await ctx.contextualizeBatchViaBatchApi(items(1));

            expect(result).toEqual(['plain']);
        });
    });
});
