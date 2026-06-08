import { AstCodeSplitter } from './ast-splitter';

describe('AstCodeSplitter oversized-node splitting', () => {
    // A class whose body comfortably exceeds a small chunkSize, with several
    // distinct methods so we can detect mid-method cuts.
    const code = `export class Calculator {
  addNumbers(a: number, b: number): number {
    const sumResult = a + b;
    return sumResult;
  }
  subtractNumbers(a: number, b: number): number {
    const diffResult = a - b;
    return diffResult;
  }
  multiplyNumbers(a: number, b: number): number {
    const productResult = a * b;
    return productResult;
  }
}`;

    it('cuts an oversized class on method boundaries, never mid-method', async () => {
        // Small chunkSize forces the whole-class node to be re-split; overlap 0
        // keeps chunk content clean for assertions.
        const splitter = new AstCodeSplitter(120, 0);
        const chunks = await splitter.split(code, 'typescript', 'calculator.ts');

        // Each method's unique body line implies its signature is in the SAME chunk.
        // With the old character-based splitter, at least one class fragment would
        // contain a body line without its method signature.
        const cases: Array<[string, string]> = [
            ['const sumResult', 'addNumbers'],
            ['const diffResult', 'subtractNumbers'],
            ['const productResult', 'multiplyNumbers'],
        ];

        for (const chunk of chunks) {
            for (const [bodyMarker, signature] of cases) {
                if (chunk.content.includes(bodyMarker)) {
                    expect(chunk.content).toContain(signature);
                }
            }
        }
    });

    it('keeps each method body intact somewhere in the output', async () => {
        const splitter = new AstCodeSplitter(120, 0);
        const chunks = await splitter.split(code, 'typescript', 'calculator.ts');

        const methodBodies = [
            'const sumResult = a + b;',
            'const diffResult = a - b;',
            'const productResult = a * b;',
        ];
        for (const body of methodBodies) {
            expect(chunks.some(c => c.content.includes(body))).toBe(true);
        }
    });

    it('does not add cross-chunk overlap to AST-aligned chunks', async () => {
        // Non-zero overlap configured. The standalone method_definition chunk must
        // begin exactly at its signature; with overlap enabled it would instead be
        // prefixed by the tail of the previous chunk.
        const splitter = new AstCodeSplitter(120, 50);
        const chunks = await splitter.split(code, 'typescript', 'calculator.ts');

        expect(chunks.some(c => c.content.trim().startsWith('addNumbers'))).toBe(true);
    });

    it('folds header keywords forward instead of emitting tiny fragments', async () => {
        const splitter = new AstCodeSplitter(120, 0);
        const chunks = await splitter.split(code, 'typescript', 'calculator.ts');

        // No degenerate fragment chunks like "export" or a bare "class Calculator".
        for (const chunk of chunks) {
            expect(chunk.content.trim()).not.toMatch(/^(export|class\s+\w+)$/);
        }
        // The class signature is folded into the first sub-chunk alongside code.
        expect(chunks.some(c => c.content.trim().startsWith('class Calculator {'))).toBe(true);
    });

    it('returns line ranges within the source bounds', async () => {
        const splitter = new AstCodeSplitter(120, 0);
        const chunks = await splitter.split(code, 'typescript', 'calculator.ts');
        const totalLines = code.split('\n').length;

        for (const chunk of chunks) {
            expect(chunk.metadata.startLine).toBeGreaterThanOrEqual(1);
            expect(chunk.metadata.endLine).toBeLessThanOrEqual(totalLines);
            expect(chunk.metadata.endLine).toBeGreaterThanOrEqual(chunk.metadata.startLine);
        }
    });
});
