#!/usr/bin/env node

// CRITICAL: Redirect console outputs to stderr IMMEDIATELY to avoid interfering with MCP JSON protocol
// Only MCP protocol messages should go to stdout
const originalConsoleLog = console.log;
const originalConsoleWarn = console.warn;

console.log = (...args: any[]) => {
  process.stderr.write("[LOG] " + args.join(" ") + "\n");
};

console.warn = (...args: any[]) => {
  process.stderr.write("[WARN] " + args.join(" ") + "\n");
};

// console.error already goes to stderr by default

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { Context } from "@zilliz/claude-context-core";
import { MilvusVectorDatabase } from "@zilliz/claude-context-core";

// Import our modular components
import {
  createMcpConfig,
  logConfigurationSummary,
  showHelpMessage,
  ContextMcpConfig,
} from "./config.js";
import {
  createEmbeddingInstance,
  logEmbeddingProviderInfo,
} from "./embedding.js";
import { SnapshotManager } from "./snapshot.js";
import { SyncManager } from "./sync.js";
import { ToolHandlers } from "./handlers.js";

// Server-level guidance surfaced to the agent in the MCP `initialize` response.
// This is the one place to frame the whole server, so it is aimed squarely at
// the agent's default "grep/glob first" bias: explain when semantic retrieval
// is the better first move, and where lexical search still wins.
const SERVER_INSTRUCTIONS = `
This server provides SEMANTIC code search over a vector index of the user's codebase: it retrieves code by *meaning*, not just literal text.

Most coding agents are trained to reach for grep/glob/find first. For finding code by *concept or behavior* — "where is auth enforced", "how are webhook retries handled" — that instinct is counterproductive: it forces many noisy lexical passes over the repo and burns context, and still misses code that doesn't contain your guessed keyword. Prefer \`search_code\` as your FIRST move whenever you know *what the code does* but not its exact name or location. Retrieval is hybrid (keyword + embeddings over AST-aware chunks), so vague phrasing works and exact terms still rank well.

This does not replace grep — it complements it. Keep using grep for literal strings you already know (an error message, a config key, a symbol you can name exactly) and for file types this server does not index (most config/data files). The best results come from leading with semantic search for conceptual questions and falling back to grep for exact matches.

Preconditions: \`search_code\` requires the codebase to be indexed. If a search returns nothing, call \`get_indexing_status\` before concluding the code is absent — the index may be missing or stale rather than the code missing.
`.trim();

class ContextMcpServer {
  private server: Server;
  private context: Context;
  private snapshotManager: SnapshotManager;
  private syncManager: SyncManager;
  private toolHandlers: ToolHandlers;

  constructor(config: ContextMcpConfig) {
    // Initialize MCP server
    this.server = new Server(
      {
        name: config.name,
        version: config.version,
      },
      {
        capabilities: {
          tools: {},
        },
        instructions: SERVER_INSTRUCTIONS,
      },
    );

    // Initialize embedding provider
    console.log(
      `[EMBEDDING] Initializing embedding provider: ${config.embeddingProvider}`,
    );
    console.log(`[EMBEDDING] Using model: ${config.embeddingModel}`);

    const embedding = createEmbeddingInstance(config);
    logEmbeddingProviderInfo(config, embedding);

    // Initialize vector database
    const vectorDatabase = new MilvusVectorDatabase({
      address: config.milvusAddress,
      ...(config.milvusToken && { token: config.milvusToken }),
    });

    // Initialize Claude Context
    this.context = new Context({
      embedding,
      vectorDatabase,
      collectionNameOverride: config.collectionNameOverride,
    });

    // Initialize managers
    this.snapshotManager = new SnapshotManager();
    this.syncManager = new SyncManager(this.context, this.snapshotManager);
    this.toolHandlers = new ToolHandlers(this.context, this.snapshotManager);

    // Load existing codebase snapshot on startup
    this.snapshotManager.loadCodebaseSnapshot();

    this.setupTools();
  }

  private setupTools() {
    const index_description = `
Index (or re-index) a codebase so it can be semantically searched with \`search_code\`. Builds AST-aware code chunks and embeddings stored in a vector database.

When to use:
- Before the first \`search_code\` on a codebase, when \`get_indexing_status\` shows it is not indexed, or when \`search_code\` reports the codebase isn't indexed.
- With \`force: true\` to rebuild an existing index after the code has changed substantially. A force re-index overwrites the existing index and re-embeds everything, so you MUST prompt the user to confirm before forcing.

Notes:
- You MUST provide an ABSOLUTE path to the target codebase.
- Indexing runs in the background and can take a while on large repos; poll \`get_indexing_status\` for progress.
- The defaults (\`splitter: 'ast'\`, standard extensions and ignore patterns) are right for almost all cases. Only pass \`customExtensions\` / \`ignorePatterns\` when the user explicitly asks for them.
`;

    const search_description = `
Semantic (meaning-based) search over the indexed codebase. Finds relevant code by *what it does*, not by literal text — so it works when you know the behavior but not the symbol name, file, or exact wording.

USE THIS FIRST for conceptual or exploratory questions, in preference to grep/glob. The default instinct to start with lexical search is the wrong move here: it takes many noisy passes, misses code that doesn't contain your guessed keyword, and burns context. One semantic query usually returns the right chunk directly. Typical triggers:
- "Where is X handled / enforced / configured?" (e.g. rate limiting, authentication, retries)
- "How does Y work?" — understanding an unfamiliar area before changing it
- Gathering all code related to a feature or concept for refactoring or review
- Locating likely bug sites or duplicated logic when you don't know the exact names

Retrieval is hybrid (keyword + vector over AST-aware chunks), so vague phrasing works AND exact identifiers still rank well. Returns ranked code snippets, each with its file path and line range.

Do NOT use this for:
- An exact string you already know — an error message, a config key, or a symbol you can name precisely. grep is faster and exact for those.
- Config/data files (e.g. .env, .ini, .xml). These are typically NOT indexed; use grep for them.

Preconditions:
- You MUST provide an ABSOLUTE path.
- The codebase must be indexed. If results are empty or look stale, call \`get_indexing_status\` (and \`index_codebase\` if needed) before concluding the code does not exist.
`;

    // Define available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: [
          {
            name: "index_codebase",
            description: index_description,
            inputSchema: {
              type: "object",
              properties: {
                path: {
                  type: "string",
                  description: `ABSOLUTE path to the codebase directory to index.`,
                },
                force: {
                  type: "boolean",
                  description: "Force re-indexing even if already indexed",
                  default: false,
                },
                splitter: {
                  type: "string",
                  description:
                    "Code splitter to use: 'ast' for syntax-aware splitting with automatic fallback, 'langchain' for character-based splitting",
                  enum: ["ast", "langchain"],
                  default: "ast",
                },
                customExtensions: {
                  type: "array",
                  items: {
                    type: "string",
                  },
                  description:
                    "Optional: Additional file extensions to include beyond defaults (e.g., ['.vue', '.svelte', '.astro']). Extensions should include the dot prefix or will be automatically added",
                  default: [],
                },
                ignorePatterns: {
                  type: "array",
                  items: {
                    type: "string",
                  },
                  description:
                    "Optional: Additional ignore patterns to exclude specific files/directories beyond defaults. Only include this parameter if the user explicitly requests custom ignore patterns (e.g., ['static/**', '*.tmp', 'private/**'])",
                  default: [],
                },
              },
              required: ["path"],
            },
          },
          {
            name: "search_code",
            description: search_description,
            inputSchema: {
              type: "object",
              properties: {
                path: {
                  type: "string",
                  description: `ABSOLUTE path to the codebase directory to search in.`,
                },
                query: {
                  type: "string",
                  description:
                    "Natural-language description of the code you want, phrased by behavior or intent rather than an exact symbol name (e.g. 'where webhook signatures are verified', 'retry logic for failed uploads'). Full sentences and domain terms both help; you do not need to know the function or file name.",
                },
                limit: {
                  type: "number",
                  description: "Maximum number of results to return",
                  default: 10,
                  maximum: 50,
                },
                extensionFilter: {
                  type: "array",
                  items: {
                    type: "string",
                  },
                  description:
                    "Optional: List of file extensions to filter results. (e.g., ['.ts','.py']).",
                  default: [],
                },
              },
              required: ["path", "query"],
            },
          },
          {
            name: "clear_index",
            description: `Delete the search index for a codebase (removes its vector collection and local snapshot). This is irreversible: \`search_code\` will not work for this path until it is re-indexed with \`index_codebase\`. Use only when explicitly asked to reset/clear the index, or to recover from a corrupted index. You MUST provide an ABSOLUTE path.`,
            inputSchema: {
              type: "object",
              properties: {
                path: {
                  type: "string",
                  description: `ABSOLUTE path to the codebase directory to clear.`,
                },
              },
              required: ["path"],
            },
          },
          {
            name: "get_indexing_status",
            description: `Check whether a codebase is indexed and ready for \`search_code\`. Returns indexing progress (a percentage while building), completion status, file/chunk counts, and when the index was last updated. Use this to decide your next move: if \`search_code\` returns nothing or looks stale, check here BEFORE concluding the code is absent — the index may be missing, mid-build, or out of date rather than the code not existing. If it is not indexed, call \`index_codebase\`. You MUST provide an ABSOLUTE path.`,
            inputSchema: {
              type: "object",
              properties: {
                path: {
                  type: "string",
                  description: `ABSOLUTE path to the codebase directory to check status for.`,
                },
              },
              required: ["path"],
            },
          },
        ],
      };
    });

    // Handle tool execution
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      switch (name) {
        case "index_codebase":
          return await this.toolHandlers.handleIndexCodebase(args);
        case "search_code":
          return await this.toolHandlers.handleSearchCode(args);
        case "clear_index":
          return await this.toolHandlers.handleClearIndex(args);
        case "get_indexing_status":
          return await this.toolHandlers.handleGetIndexingStatus(args);

        default:
          throw new Error(`Unknown tool: ${name}`);
      }
    });
  }

  async start() {
    console.log("[SYNC-DEBUG] MCP server start() method called");
    console.log("Starting Context MCP server...");

    // One-shot startup healing for legacy 0/0+completed snapshot entries
    // left over from pre-fix MCP versions. Runs before the transport accepts
    // requests so clients never observe the poisoning state. See Issue #295.
    await this.toolHandlers.validateLegacyZeroEntries();

    const transport = new StdioServerTransport();
    console.log(
      "[SYNC-DEBUG] StdioServerTransport created, attempting server connection...",
    );

    await this.server.connect(transport);
    console.log("MCP server started and listening on stdio.");
    console.log("[SYNC-DEBUG] Server connection established successfully");

    // Start background sync after server is connected
    console.log("[SYNC-DEBUG] Initializing background sync...");
    this.syncManager.startBackgroundSync();
    console.log("[SYNC-DEBUG] MCP server initialization complete");
  }
}

// Main execution
async function main() {
  // Parse command line arguments
  const args = process.argv.slice(2);

  // Show help if requested
  if (args.includes("--help") || args.includes("-h")) {
    showHelpMessage();
    process.exit(0);
  }

  // Create configuration
  const config = createMcpConfig();
  logConfigurationSummary(config);

  const server = new ContextMcpServer(config);
  await server.start();
}

// Handle graceful shutdown
process.on("SIGINT", () => {
  console.error("Received SIGINT, shutting down gracefully...");
  process.exit(0);
});

process.on("SIGTERM", () => {
  console.error("Received SIGTERM, shutting down gracefully...");
  process.exit(0);
});

// Always start the server - this is designed to be the main entry point
main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
