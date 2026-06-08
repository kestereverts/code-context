import * as fs from 'fs/promises';
import * as os from 'os';
import * as path from 'path';
import { Context } from './context';
import { Embedding, EmbeddingVector } from './embedding';
import { Splitter, CodeChunk } from './splitter';
import { VectorDatabase } from './vectordb';

// Captures the exact texts handed to embedBatch so we can assert the context
// blurb was prepended to the embedding input.
class RecordingEmbedding extends Embedding {
    protected maxTokens = 8192;
    public embedded: string[] = [];

    async detectDimension(): Promise<number> { return 3; }
    async embed(text: string): Promise<EmbeddingVector> {
        this.embedded.push(text);
        return { vector: [1, 0, 0], dimension: 3 };
    }
    async embedBatch(texts: string[]): Promise<EmbeddingVector[]> {
        this.embedded.push(...texts);
        return texts.map(() => ({ vector: [1, 0, 0], dimension: 3 }));
    }
    getDimension(): number { return 3; }
    getProvider(): string { return 'test'; }
}

class OneChunkPerFileSplitter implements Splitter {
    async split(code: string, language: string, filePath?: string): Promise<CodeChunk[]> {
        return [{
            content: code,
            metadata: { startLine: 1, endLine: code.split('\n').length, language, filePath },
        }];
    }
    setChunkSize(): void { }
    setChunkOverlap(): void { }
}

const createVectorDatabase = (): jest.Mocked<VectorDatabase> => ({
    createCollection: jest.fn().mockResolvedValue(undefined),
    createHybridCollection: jest.fn().mockResolvedValue(undefined),
    dropCollection: jest.fn().mockResolvedValue(undefined),
    hasCollection: jest.fn().mockResolvedValue(false),
    listCollections: jest.fn().mockResolvedValue([]),
    insert: jest.fn().mockResolvedValue(undefined),
    insertHybrid: jest.fn().mockResolvedValue(undefined),
    search: jest.fn().mockResolvedValue([]),
    hybridSearch: jest.fn().mockResolvedValue([]),
    delete: jest.fn().mockResolvedValue(undefined),
    query: jest.fn().mockResolvedValue([]),
    getCollectionDescription: jest.fn().mockResolvedValue(''),
    checkCollectionLimit: jest.fn().mockResolvedValue(true),
    getCollectionRowCount: jest.fn().mockResolvedValue(0),
});

describe('Context batch contextualization pre-pass', () => {
    let tempRoot: string;
    let originalHome: string | undefined;
    let originalHybridMode: string | undefined;

    beforeEach(async () => {
        tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'claude-context-batchctx-'));
        const homeDir = path.join(tempRoot, 'home');
        await fs.mkdir(homeDir, { recursive: true });
        originalHome = process.env.HOME;
        originalHybridMode = process.env.HYBRID_MODE;
        process.env.HOME = homeDir;
        process.env.HYBRID_MODE = 'false';
    });

    afterEach(async () => {
        if (originalHome === undefined) delete process.env.HOME; else process.env.HOME = originalHome;
        if (originalHybridMode === undefined) delete process.env.HYBRID_MODE; else process.env.HYBRID_MODE = originalHybridMode;
        await fs.rm(tempRoot, { recursive: true, force: true });
    });

    it('runs one batch job and applies precomputed contexts to embeddings + metadata', async () => {
        const project = path.join(tempRoot, 'project');
        await fs.mkdir(project);
        await fs.writeFile(path.join(project, 'a.ts'), 'const a = 1;');
        await fs.writeFile(path.join(project, 'b.ts'), 'const b = 2;');

        // Fake contextualizer: echoes a deterministic blurb per chunk.
        const contextualizeBatchViaBatchApi = jest.fn(async (items: Array<{ chunkContent: string }>) =>
            items.map(it => `CTX:${it.chunkContent}`)
        );
        const fakeContextualizer = { contextualizeBatchViaBatchApi } as any;

        const embedding = new RecordingEmbedding();
        const vectorDatabase = createVectorDatabase();
        const context = new Context({
            embedding,
            vectorDatabase,
            codeSplitter: new OneChunkPerFileSplitter(),
            contextualizer: fakeContextualizer,
            useBatchContextualization: true,
        });

        await context.indexCodebase(project);

        // Exactly one batch job covering both chunks.
        expect(contextualizeBatchViaBatchApi).toHaveBeenCalledTimes(1);
        expect(contextualizeBatchViaBatchApi.mock.calls[0][0]).toHaveLength(2);

        // Embedding input = context blurb prepended to the verbatim chunk.
        expect(embedding.embedded).toContain('CTX:const a = 1;\n\nconst a = 1;');
        expect(embedding.embedded).toContain('CTX:const b = 2;\n\nconst b = 2;');

        // Stored content stays verbatim; the blurb lands in metadata.context.
        const docs = vectorDatabase.insert.mock.calls.flatMap(([, d]) => d);
        expect(docs).toHaveLength(2);
        for (const doc of docs) {
            expect(doc.content).not.toContain('CTX:');
            expect((doc.metadata as any).context).toBe(`CTX:${doc.content}`);
        }
    });

    it('reports smooth progress across the batch wait instead of freezing', async () => {
        const project = path.join(tempRoot, 'project');
        await fs.mkdir(project);
        await fs.writeFile(path.join(project, 'a.ts'), 'const a = 1;');
        await fs.writeFile(path.join(project, 'b.ts'), 'const b = 2;');

        // Fake contextualizer that emits heartbeats with growing elapsed time,
        // exercising the Context's elapsed-time creep.
        const contextualizeBatchViaBatchApi = jest.fn(async (
            items: Array<{ chunkContent: string }>,
            onProgress?: (p: { jobNum: number; jobTotal: number; state: string; elapsedMs: number }) => void,
        ) => {
            onProgress?.({ jobNum: 1, jobTotal: 1, state: 'JOB_STATE_PENDING', elapsedMs: 1000 });
            onProgress?.({ jobNum: 1, jobTotal: 1, state: 'JOB_STATE_RUNNING', elapsedMs: 60_000 });
            onProgress?.({ jobNum: 1, jobTotal: 1, state: 'JOB_STATE_RUNNING', elapsedMs: 180_000 });
            return items.map(it => `CTX:${it.chunkContent}`);
        });

        const context = new Context({
            embedding: new RecordingEmbedding(),
            vectorDatabase: createVectorDatabase(),
            codeSplitter: new OneChunkPerFileSplitter(),
            contextualizer: { contextualizeBatchViaBatchApi } as any,
            useBatchContextualization: true,
        });

        const pcts: number[] = [];
        await context.indexCodebase(project, (p) => pcts.push(p.percentage));

        // Not frozen: progress moves through the middle of the wait band (15–75).
        expect(pcts.some(p => p > 15 && p < 75)).toBe(true);
        // The wait creep is monotonic across the three heartbeats.
        // The embed pass takes the top slice and the run reaches completion.
        expect(pcts.some(p => p >= 75 && p < 100)).toBe(true);
        expect(Math.max(...pcts)).toBe(100);
        // Overall progress never goes backwards.
        for (let i = 1; i < pcts.length; i++) {
            expect(pcts[i]).toBeGreaterThanOrEqual(pcts[i - 1]);
        }
    });

    it('does not run the pre-pass when batch mode is disabled', async () => {
        const project = path.join(tempRoot, 'project');
        await fs.mkdir(project);
        await fs.writeFile(path.join(project, 'a.ts'), 'const a = 1;');

        const contextualizeBatchViaBatchApi = jest.fn();
        const fakeContextualizer = { contextualizeBatchViaBatchApi } as any;

        const context = new Context({
            embedding: new RecordingEmbedding(),
            vectorDatabase: createVectorDatabase(),
            codeSplitter: new OneChunkPerFileSplitter(),
            contextualizer: fakeContextualizer,
            useBatchContextualization: false,
        });

        await context.indexCodebase(project);

        expect(contextualizeBatchViaBatchApi).not.toHaveBeenCalled();
    });
});
