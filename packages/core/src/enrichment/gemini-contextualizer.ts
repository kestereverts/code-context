import { GoogleGenAI } from "@google/genai";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";

/**
 * Input for contextualizing a single chunk: the chunk itself plus the
 * full text of the file it was extracted from, used to situate it.
 */
export interface ContextualizeItem {
  chunkContent: string;
  fileContent: string;
  relativePath: string;
}

/** Liveness heartbeat emitted while waiting on an async batch job. */
export interface BatchProgress {
  /** 1-based index of the current inline job. */
  jobNum: number;
  /** Total number of inline jobs in this batch run. */
  jobTotal: number;
  /** Latest JobState reported by the service (e.g. JOB_STATE_RUNNING). */
  state: string;
  /** Milliseconds elapsed since this job started polling. */
  elapsedMs: number;
}

export interface GeminiContextualizerConfig {
  apiKey: string;
  /** LLM used to generate chunk context. Defaults to 'gemini-3.1-flash-lite'. */
  model?: string;
  /** Optional custom API endpoint URL (mirrors GeminiEmbedding). */
  baseURL?: string;
  /** Max concurrent LLM calls during a batch. Defaults to 5. */
  maxConcurrency?: number;
  /**
   * Cap on how many characters of the enclosing file are sent to the LLM.
   * Protects against blowing the context window / cost on huge files.
   * Defaults to 60_000 (~15K tokens).
   */
  maxFileChars?: number;
  /** Poll interval (ms) while waiting on a batch job. Defaults to 30_000. */
  batchPollIntervalMs?: number;
  /** Max time (ms) to wait for a batch job before giving up. Defaults to 3_600_000 (1h). */
  batchMaxWaitMs?: number;
  /** Max serialized bytes per JSONL batch file (Gemini file limit is 2GB). Defaults to 100_000_000. */
  batchMaxBytes?: number;
}

const DEFAULT_MODEL = "gemini-2.5-flash-lite";
const DEFAULT_MAX_CONCURRENCY = 5;
const DEFAULT_MAX_FILE_CHARS = 60_000;
const DEFAULT_BATCH_POLL_INTERVAL_MS = 30_000;
const DEFAULT_BATCH_MAX_WAIT_MS = 3_600_000;
const DEFAULT_BATCH_MAX_BYTES = 100_000_000;

// Terminal states for a batch job. The Gemini API is inconsistent about the
// prefix (we've observed both JOB_STATE_* and BATCH_STATE_*), so accept both.
const BATCH_COMPLETED_STATES = new Set([
  "JOB_STATE_SUCCEEDED",
  "JOB_STATE_FAILED",
  "JOB_STATE_CANCELLED",
  "JOB_STATE_EXPIRED",
  "BATCH_STATE_SUCCEEDED",
  "BATCH_STATE_FAILED",
  "BATCH_STATE_CANCELLED",
  "BATCH_STATE_EXPIRED",
]);
const BATCH_SUCCESS_STATES = new Set([
  "JOB_STATE_SUCCEEDED",
  "BATCH_STATE_SUCCEEDED",
]);

/**
 * Generates short, natural-language context blurbs for code chunks using a
 * Gemini chat model, implementing the indexing side of "contextual retrieval"
 * (https://www.anthropic.com/news/contextual-retrieval).
 *
 * The blurb is meant to be prepended to a chunk *before embedding* so the dense
 * vector captures how the chunk fits into its file (callers, purpose, role),
 * closing the gap between conceptual queries and bare code.
 *
 * NOTE: This currently enriches only the dense (embedding) stream. The BM25
 * sparse stream still indexes the verbatim chunk content. Extending the benefit
 * to BM25 would require storing the enriched text in a separate searchable
 * field on the Milvus schema — left as a follow-up.
 */
export class GeminiContextualizer {
  private client: GoogleGenAI;
  private model: string;
  private maxConcurrency: number;
  private maxFileChars: number;
  private batchPollIntervalMs: number;
  private batchMaxWaitMs: number;
  private batchMaxBytes: number;

  constructor(config: GeminiContextualizerConfig) {
    this.client = new GoogleGenAI({
      apiKey: config.apiKey,
      ...(config.baseURL && {
        httpOptions: { baseUrl: config.baseURL },
      }),
    });
    this.model = config.model || DEFAULT_MODEL;
    this.maxConcurrency = config.maxConcurrency ?? DEFAULT_MAX_CONCURRENCY;
    this.maxFileChars = config.maxFileChars ?? DEFAULT_MAX_FILE_CHARS;
    this.batchPollIntervalMs =
      config.batchPollIntervalMs ?? DEFAULT_BATCH_POLL_INTERVAL_MS;
    this.batchMaxWaitMs = config.batchMaxWaitMs ?? DEFAULT_BATCH_MAX_WAIT_MS;
    this.batchMaxBytes = config.batchMaxBytes ?? DEFAULT_BATCH_MAX_BYTES;
  }

  getModel(): string {
    return this.model;
  }

  /**
   * Generate a context blurb for one chunk. Returns an empty string on any
   * failure so indexing can proceed with the raw chunk.
   */
  async contextualize(item: ContextualizeItem): Promise<string> {
    const prompt = this.promptFor(item);

    try {
      const response = await this.client.models.generateContent({
        model: this.model,
        contents: prompt,
        config: {
          temperature: 0,
          maxOutputTokens: 200,
        },
      });
      return (response.text || "").trim();
    } catch (error) {
      console.warn(
        `[GeminiContextualizer] ⚠️  Failed to contextualize chunk from ${item.relativePath}: ${error instanceof Error ? error.message : "Unknown error"}`,
      );
      return "";
    }
  }

  /**
   * Contextualize many chunks with bounded concurrency. Order of the returned
   * array matches the input. Individual failures yield '' (no context).
   */
  async contextualizeBatch(items: ContextualizeItem[]): Promise<string[]> {
    const results: string[] = new Array(items.length).fill("");
    let cursor = 0;

    const worker = async (): Promise<void> => {
      while (cursor < items.length) {
        const index = cursor++;
        results[index] = await this.contextualize(items[index]);
      }
    };

    const workers = Array.from(
      { length: Math.min(this.maxConcurrency, items.length) },
      () => worker(),
    );
    await Promise.all(workers);
    return results;
  }

  /**
   * Contextualize many chunks via the Gemini Batch API (~50% cheaper, async).
   *
   * Uses the file-based (JSONL) batch path with a stable per-request `key`, and
   * matches results BY KEY rather than by position. This is deliberate: inline
   * batch responses carry no key, so the only correlation is array position —
   * and the service can drop/reorder responses, which silently attaches a blurb
   * to the wrong chunk. Key-based matching makes a missing/dropped result yield
   * NO context for that chunk, never another chunk's context.
   *
   * Returns blurbs aligned to the input array. Any job failure or timeout yields
   * '' for that job's chunks so indexing proceeds.
   */
  async contextualizeBatchViaBatchApi(
    items: ContextualizeItem[],
    onProgress?: (progress: BatchProgress) => void,
  ): Promise<string[]> {
    const results: string[] = new Array(items.length).fill("");
    if (items.length === 0) return results;

    const entries = items.map((item, index) => {
      const key = `req-${index}`;
      return {
        index,
        key,
        line: JSON.stringify({ key, request: this.buildRequest(item) }),
      };
    });

    const groups = this.packGroups(entries);
    console.log(
      `[GeminiContextualizer] 📦 Batch contextualization: ${items.length} chunks across ${groups.length} file job(s) (model: ${this.model})`,
    );

    for (let g = 0; g < groups.length; g++) {
      const group = groups[g];
      try {
        const byKey = await this.runFileBatchJob(
          group,
          g + 1,
          groups.length,
          onProgress,
        );
        for (const entry of group) {
          const text = byKey.get(entry.key);
          if (text) results[entry.index] = text;
        }
      } catch (error) {
        console.warn(
          `[GeminiContextualizer] ⚠️  Batch job ${g + 1}/${groups.length} failed; leaving ${group.length} chunks without context: ${error instanceof Error ? error.message : "Unknown error"}`,
        );
      }
    }
    return results;
  }

  /** Pack JSONL request lines into groups that each stay under the file byte cap. */
  private packGroups(
    entries: Array<{ index: number; key: string; line: string }>,
  ): Array<Array<{ index: number; key: string; line: string }>> {
    const groups: Array<Array<{ index: number; key: string; line: string }>> =
      [];
    let current: Array<{ index: number; key: string; line: string }> = [];
    let bytes = 0;

    for (const entry of entries) {
      const size = entry.line.length + 1; // + newline
      if (current.length > 0 && bytes + size > this.batchMaxBytes) {
        groups.push(current);
        current = [];
        bytes = 0;
      }
      current.push(entry);
      bytes += size;
    }
    if (current.length > 0) groups.push(current);
    return groups;
  }

  /** Submit one file-based batch job, wait for it, and return a key→text map. */
  private async runFileBatchJob(
    group: Array<{ index: number; key: string; line: string }>,
    jobNum: number,
    jobTotal: number,
    onProgress?: (progress: BatchProgress) => void,
  ): Promise<Map<string, string>> {
    const jsonl = group.map((e) => e.line).join("\n") + "\n";

    // Upload the JSONL input via a temp file (SDK accepts a file path).
    const inputPath = path.join(
      os.tmpdir(),
      `code-context-batch-in-${Date.now()}-${jobNum}.jsonl`,
    );
    await fs.promises.writeFile(inputPath, jsonl, "utf-8");
    let uploadedName: string | undefined;
    try {
      const uploaded = await this.client.files.upload({
        file: inputPath,
        config: { mimeType: "jsonl" },
      });
      uploadedName = uploaded.name;
    } finally {
      fs.promises.unlink(inputPath).catch(() => {});
    }
    if (!uploadedName)
      throw new Error("Uploaded batch input file has no resource name");

    const created = await this.client.batches.create({
      model: this.model,
      src: uploadedName,
      config: {
        displayName: `code-context-contextualizer-${Date.now()}-${jobNum}`,
      },
    });
    const name = created.name;
    if (!name) throw new Error("Batch job created without a resource name");

    const job = await this.pollJob(name, jobNum, jobTotal, onProgress);
    if (!BATCH_SUCCESS_STATES.has(job.state as string)) {
      const detail = job.error ? `: ${JSON.stringify(job.error)}` : "";
      throw new Error(`Batch job ${name} ended in state ${job.state}${detail}`);
    }

    // Results may be returned inline (no key, but order is the only option) or
    // as a downloadable file (keyed — preferred). Prefer the file.
    const resultFile = job.dest?.fileName;
    if (!resultFile) {
      throw new Error(
        `Batch job ${name} succeeded but produced no result file`,
      );
    }

    // NOTE: the SDK's files.download() resolves BEFORE the write to disk
    // finishes (the stream pipe / fs.writeFile callback are not awaited), so we
    // must wait for the file to finish writing before reading it.
    const outputPath = path.join(
      os.tmpdir(),
      `code-context-batch-out-${Date.now()}-${jobNum}.jsonl`,
    );
    await this.client.files.download({
      file: resultFile,
      downloadPath: outputPath,
    });
    let content: string;
    try {
      await this.waitForFileSettled(outputPath);
      content = await fs.promises.readFile(outputPath, "utf-8");
    } finally {
      fs.promises.unlink(outputPath).catch(() => {});
    }

    return this.parseResultsByKey(content);
  }

  /** Parse a result JSONL file into a key→text map, skipping errored lines. */
  private parseResultsByKey(jsonl: string): Map<string, string> {
    const map = new Map<string, string>();
    for (const raw of jsonl.split("\n")) {
      const line = raw.trim();
      if (!line) continue;
      let obj: any;
      try {
        obj = JSON.parse(line);
      } catch {
        continue;
      }
      const key = obj.key;
      if (typeof key !== "string") continue;
      if (obj.error || obj.response?.error) {
        console.warn(
          `[GeminiContextualizer] ⚠️  Batch result for ${key} reported an error; no context.`,
        );
        continue;
      }
      const text = this.extractText(obj.response);
      if (text) map.set(key, text);
    }
    return map;
  }

  /** Poll a batch job until it reaches a terminal state or the deadline passes. */
  private async pollJob(
    name: string,
    jobNum: number,
    jobTotal: number,
    onProgress?: (progress: BatchProgress) => void,
  ): Promise<any> {
    const start = Date.now();
    const deadline = start + this.batchMaxWaitMs;
    let job: any = await this.client.batches.get({ name });

    while (!BATCH_COMPLETED_STATES.has(job.state as string)) {
      // Heartbeat: lets callers advance their own status/timestamp so observers
      // can tell "alive, waiting on async job" apart from "hung".
      onProgress?.({
        jobNum,
        jobTotal,
        state: String(job.state),
        elapsedMs: Date.now() - start,
      });

      if (Date.now() >= deadline) {
        // Best-effort cancel so we stop paying for an abandoned job.
        try {
          await this.client.batches.cancel({ name });
        } catch {
          /* ignore */
        }
        throw new Error(
          `Batch job ${name} timed out after ${this.batchMaxWaitMs}ms (last state: ${job.state})`,
        );
      }
      console.log(
        `[GeminiContextualizer] ⏳ Batch job ${jobNum}/${jobTotal} state: ${job.state}; re-checking in ${this.batchPollIntervalMs}ms...`,
      );
      await this.sleep(this.batchPollIntervalMs);
      job = await this.client.batches.get({ name });
    }
    return job;
  }

  /** Extract text from a GenerateContentResponse, tolerating plain-JSON shapes. */
  private extractText(response: any): string {
    if (!response) return "";
    if (typeof response.text === "string") return response.text.trim();
    const parts = response.candidates?.[0]?.content?.parts;
    if (Array.isArray(parts)) {
      return parts
        .map((p: any) => p?.text || "")
        .join("")
        .trim();
    }
    return "";
  }

  /** Build a JSONL batch `request` (raw GenerateContentRequest shape) for an item. */
  private buildRequest(item: ContextualizeItem): any {
    return {
      // System guidance must go in systemInstruction — Gemini's `contents` only
      // accepts the roles "user" and "model"; a "system" role is rejected (400).
      systemInstruction: {
        parts: [
          {
            text: `You are a semantic indexing assistant for codebases who describes code chunks in the context of their enclosing file. Your writing style is concise, factual, terse, but descriptive. You write identifiers in backticks.`,
          },
        ],
      },
      contents: [
        { role: "user", parts: [{ text: this.promptFor(item) }] },
      ],
      generationConfig: { temperature: 0, maxOutputTokens: 200 },
    };
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  /**
   * Wait until a file exists and its size stops changing — needed because the
   * SDK's files.download() returns before the on-disk write completes. Resolves
   * once the size is non-zero and stable across two checks, or throws on timeout.
   */
  private async waitForFileSettled(
    filePath: string,
    timeoutMs = 30_000,
  ): Promise<void> {
    const deadline = Date.now() + timeoutMs;
    let lastSize = -1;
    let stableSince = 0;
    const STABLE_MS = 150;
    const POLL_MS = 50;

    while (Date.now() < deadline) {
      let size: number | undefined;
      try {
        size = (await fs.promises.stat(filePath)).size;
      } catch {
        size = undefined; // not created yet
      }
      if (size !== undefined && size > 0) {
        if (size === lastSize) {
          if (Date.now() - stableSince >= STABLE_MS) return;
        } else {
          lastSize = size;
          stableSince = Date.now();
        }
      }
      await this.sleep(POLL_MS);
    }
    throw new Error(
      `Timed out waiting for downloaded result file to finish writing: ${filePath}`,
    );
  }

  /** Truncate the enclosing file (if needed) and build the contextualization prompt. */
  private promptFor(item: ContextualizeItem): string {
    const fileContent =
      item.fileContent.length > this.maxFileChars
        ? item.fileContent.slice(0, this.maxFileChars)
        : item.fileContent;
    return this.buildPrompt(item.relativePath, fileContent, item.chunkContent);
  }

  private buildPrompt(
    relativePath: string,
    fileContent: string,
    chunkContent: string,
  ): string {
    return `<document path="${relativePath}">
${fileContent}
</document>

<chunk>
${chunkContent}
</chunk>

Write ONE TO THREE terse lines (max 75 words total) that help a semantic (contextual retrieval) search engine understand the meaning of the chunk.
State what the code in the chunk intends to do inside the context of the document.
Describe the relationship between the chunk and the rest of the document.
You may use the file path to infer the document's overall purpose inside the codebase, but be wary of ambiguity.

Rules:
- No meta-phrases. Never start with or include "This chunk", "This code", "This snippet".
- No filler adjectives or restating the obvious. Facts only.
- Write it as a standalone descriptive set of phrases, not a sentence about the chunk.
- Output only the lines, nothing else.

Example good: "Defines a concrete validation rule using a GIS library for verifying that a province has access to the area, inside a section that configures many validation rules for a geographic data model. Sits in the GeoValidation class.
Example bad: "This chunk contains a validation rule. The code in this chunk defines a validation rule using a GIS library for verifying that a province has access to the area. This line of code is part of the Geo module."`;
  }
}
