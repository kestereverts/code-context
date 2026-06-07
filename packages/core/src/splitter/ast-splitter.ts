import Parser from "tree-sitter";
import { Splitter, CodeChunk } from "./index";

// Language parsers — original languages
const JavaScript = require("tree-sitter-javascript");
const TypeScript = require("tree-sitter-typescript").typescript;
const TSX = require("tree-sitter-typescript").tsx;
const Python = require("tree-sitter-python");
const Java = require("tree-sitter-java");
const Cpp = require("tree-sitter-cpp");
const Go = require("tree-sitter-go");
const Rust = require("tree-sitter-rust");
const CSharp = require("tree-sitter-c-sharp").language;
const Scala = require("tree-sitter-scala");
// Extended web-dev languages
const Angular = require("tree-sitter-angular").language;
const Bash = require("tree-sitter-bash").language;
const CSS = require("tree-sitter-css").language;
const Elixir = require("tree-sitter-elixir").language;
const HCL = require("@tree-sitter-grammars/tree-sitter-hcl").language;
const HTML = require("tree-sitter-html").language;
const Jinja2 = require("tree-sitter-jinja2").language;
const JSON_ = require("tree-sitter-json").language;
const Kotlin = require("tree-sitter-kotlin").language;
const Markdown = require("tree-sitter-markdown").language;
const PHP = require("tree-sitter-php").php;
const Prisma = require("tree-sitter-prisma").language;
const Ruby = require("tree-sitter-ruby").language;
const SCSS = require("tree-sitter-scss").language;
const SQL = require("tree-sitter-sql").language;
const TOML = require("tree-sitter-toml").language;
const Vue = require("tree-sitter-vue").language;
const YAML = require("tree-sitter-yaml").language;

// Node types that represent logical code units
const SPLITTABLE_NODE_TYPES = {
  javascript: [
    "function_declaration",
    "arrow_function",
    "class_declaration",
    "method_definition",
    "export_statement",
  ],
  typescript: [
    "function_declaration",
    "arrow_function",
    "class_declaration",
    "method_definition",
    "export_statement",
    "interface_declaration",
    "type_alias_declaration",
  ],
  python: [
    "function_definition",
    "class_definition",
    "decorated_definition",
    "async_function_definition",
  ],
  java: [
    "method_declaration",
    "class_declaration",
    "interface_declaration",
    "constructor_declaration",
  ],
  cpp: [
    "function_definition",
    "class_specifier",
    "namespace_definition",
    "declaration",
  ],
  go: [
    "function_declaration",
    "method_declaration",
    "type_declaration",
    "var_declaration",
    "const_declaration",
  ],
  rust: [
    "function_item",
    "impl_item",
    "struct_item",
    "enum_item",
    "trait_item",
    "mod_item",
  ],
  csharp: [
    "method_declaration",
    "class_declaration",
    "interface_declaration",
    "struct_declaration",
    "enum_declaration",
  ],
  scala: [
    "method_declaration",
    "class_declaration",
    "interface_declaration",
    "constructor_declaration",
  ],
  tsx: [
    "function_declaration",
    "arrow_function",
    "class_declaration",
    "method_definition",
    "export_statement",
    "interface_declaration",
    "type_alias_declaration",
    "jsx_element",
    "jsx_self_closing_element",
  ],
  css: [
    "rule_set",
    "media_statement",
    "keyframes_statement",
    "supports_statement",
  ],
  html: [
    "element",
    "script_element",
    "style_element",
  ],
  json: [
    "pair",
  ],
  bash: [
    "function_definition",
    "if_statement",
    "for_statement",
    "while_statement",
    "case_statement",
  ],
  ruby: [
    "method",
    "class",
    "module",
    "singleton_method",
    "singleton_class",
  ],
  angular: [
    "element",
    "structural_directive",
    "script_element",
    "style_element",
  ],
  elixir: [
    "call",
    "anonymous_function",
  ],
  hcl: [
    "block",
    "attribute",
  ],
  jinja2: [
    "block_statement",
    "macro_statement",
    "for_statement",
    "if_statement",
  ],
  kotlin: [
    "function_declaration",
    "class_declaration",
    "object_declaration",
    "companion_object",
    "secondary_constructor",
  ],
  markdown: [
    "atx_heading",
    "setext_heading",
    "fenced_code_block",
    "block_quote",
  ],
  php: [
    "function_definition",
    "class_declaration",
    "method_declaration",
    "interface_declaration",
    "trait_declaration",
    "namespace_definition",
    "enum_declaration",
  ],
  prisma: [
    "model_declaration",
    "enum_declaration",
    "generator_declaration",
    "datasource_declaration",
    "type_declaration",
    "view_declaration",
  ],
  scss: [
    "rule_set",
    "mixin_statement",
    "function_statement",
    "at_rule",
  ],
  sql: [
    "select_statement",
    "create_function_statement",
    "create_table_statement",
    "create_index_statement",
    "create_type_statement",
  ],
  toml: [
    "table",
    "table_array_element",
    "pair",
  ],
  vue: [
    "element",
    "script_element",
    "style_element",
    "template_element",
  ],
  yaml: [
    "block_mapping_pair",
    "document",
  ],
};

export class AstCodeSplitter implements Splitter {
  private chunkSize: number = 2500;
  private chunkOverlap: number = 300;
  private parser: Parser;
  private langchainFallback: any; // LangChainCodeSplitter for fallback

  constructor(chunkSize?: number, chunkOverlap?: number) {
    if (chunkSize) this.chunkSize = chunkSize;
    if (chunkOverlap) this.chunkOverlap = chunkOverlap;
    this.parser = new Parser();

    // Initialize fallback splitter
    const { LangChainCodeSplitter } = require("./langchain-splitter");
    this.langchainFallback = new LangChainCodeSplitter(chunkSize, chunkOverlap);
  }

  async split(
    code: string,
    language: string,
    filePath?: string,
  ): Promise<CodeChunk[]> {
    // Check if language is supported by AST splitter
    const langConfig = this.getLanguageConfig(language);
    if (!langConfig) {
      console.log(
        `📝 Language ${language} not supported by AST, using LangChain splitter for: ${filePath || "unknown"}`,
      );
      return await this.langchainFallback.split(code, language, filePath);
    }

    try {
      console.log(
        `🌳 Using AST splitter for ${language} file: ${filePath || "unknown"}`,
      );

      this.parser.setLanguage(langConfig.parser);
      // node-tree-sitter defaults to a 32KB internal buffer and silently
      // truncates longer string inputs (tree-sitter/node-tree-sitter#250).
      // Size the buffer to the source so large files (e.g. >32KB Go files)
      // are parsed in full.
      const codeByteLength = Buffer.byteLength(code, "utf8");
      const bufferSize = Math.max(32 * 1024, codeByteLength + 1024);
      const tree = this.parser.parse(code, undefined, { bufferSize });

      if (!tree.rootNode) {
        console.warn(
          `[ASTSplitter] ⚠️  Failed to parse AST for ${language}, falling back to LangChain: ${filePath || "unknown"}`,
        );
        return await this.langchainFallback.split(code, language, filePath);
      }

      // Defensive check against silent truncation: a healthy parse must cover
      // the entire source byte range. If it doesn't, fall back so we don't
      // silently drop the tail of the file from the index.
      if (tree.rootNode.endIndex < codeByteLength) {
        console.warn(
          `[ASTSplitter] ⚠️  AST parse incomplete for ${language} (parsed ${tree.rootNode.endIndex}/${codeByteLength} bytes), falling back to LangChain: ${filePath || "unknown"}`,
        );
        return await this.langchainFallback.split(code, language, filePath);
      }

      // Extract chunks based on AST nodes
      const chunks = this.extractChunks(
        tree.rootNode,
        code,
        langConfig.nodeTypes,
        language,
        filePath,
      );

      // If chunks are too large, split them further
      const refinedChunks = await this.refineChunks(chunks, code);

      return refinedChunks;
    } catch (error) {
      console.warn(
        `[ASTSplitter] ⚠️  AST splitter failed for ${language}, falling back to LangChain: ${error}`,
      );
      return await this.langchainFallback.split(code, language, filePath);
    }
  }

  setChunkSize(chunkSize: number): void {
    this.chunkSize = chunkSize;
    this.langchainFallback.setChunkSize(chunkSize);
  }

  setChunkOverlap(chunkOverlap: number): void {
    this.chunkOverlap = chunkOverlap;
    this.langchainFallback.setChunkOverlap(chunkOverlap);
  }

  private getLanguageConfig(
    language: string,
  ): { parser: any; nodeTypes: string[] } | null {
    const langMap: Record<string, { parser: any; nodeTypes: string[] }> = {
      javascript: {
        parser: JavaScript,
        nodeTypes: SPLITTABLE_NODE_TYPES.javascript,
      },
      js: { parser: JavaScript, nodeTypes: SPLITTABLE_NODE_TYPES.javascript },
      typescript: {
        parser: TypeScript,
        nodeTypes: SPLITTABLE_NODE_TYPES.typescript,
      },
      ts: { parser: TypeScript, nodeTypes: SPLITTABLE_NODE_TYPES.typescript },
      python: { parser: Python, nodeTypes: SPLITTABLE_NODE_TYPES.python },
      py: { parser: Python, nodeTypes: SPLITTABLE_NODE_TYPES.python },
      java: { parser: Java, nodeTypes: SPLITTABLE_NODE_TYPES.java },
      cpp: { parser: Cpp, nodeTypes: SPLITTABLE_NODE_TYPES.cpp },
      "c++": { parser: Cpp, nodeTypes: SPLITTABLE_NODE_TYPES.cpp },
      c: { parser: Cpp, nodeTypes: SPLITTABLE_NODE_TYPES.cpp },
      go: { parser: Go, nodeTypes: SPLITTABLE_NODE_TYPES.go },
      rust: { parser: Rust, nodeTypes: SPLITTABLE_NODE_TYPES.rust },
      rs: { parser: Rust, nodeTypes: SPLITTABLE_NODE_TYPES.rust },
      cs: { parser: CSharp, nodeTypes: SPLITTABLE_NODE_TYPES.csharp },
      csharp: { parser: CSharp, nodeTypes: SPLITTABLE_NODE_TYPES.csharp },
      scala: { parser: Scala, nodeTypes: SPLITTABLE_NODE_TYPES.scala },
      tsx: { parser: TSX, nodeTypes: SPLITTABLE_NODE_TYPES.tsx },
      css: { parser: CSS, nodeTypes: SPLITTABLE_NODE_TYPES.css },
      html: { parser: HTML, nodeTypes: SPLITTABLE_NODE_TYPES.html },
      htm: { parser: HTML, nodeTypes: SPLITTABLE_NODE_TYPES.html },
      json: { parser: JSON_, nodeTypes: SPLITTABLE_NODE_TYPES.json },
      bash: { parser: Bash, nodeTypes: SPLITTABLE_NODE_TYPES.bash },
      sh: { parser: Bash, nodeTypes: SPLITTABLE_NODE_TYPES.bash },
      zsh: { parser: Bash, nodeTypes: SPLITTABLE_NODE_TYPES.bash },
      ruby: { parser: Ruby, nodeTypes: SPLITTABLE_NODE_TYPES.ruby },
      rb: { parser: Ruby, nodeTypes: SPLITTABLE_NODE_TYPES.ruby },
      angular: { parser: Angular, nodeTypes: SPLITTABLE_NODE_TYPES.angular },
      elixir: { parser: Elixir, nodeTypes: SPLITTABLE_NODE_TYPES.elixir },
      ex: { parser: Elixir, nodeTypes: SPLITTABLE_NODE_TYPES.elixir },
      exs: { parser: Elixir, nodeTypes: SPLITTABLE_NODE_TYPES.elixir },
      hcl: { parser: HCL, nodeTypes: SPLITTABLE_NODE_TYPES.hcl },
      terraform: { parser: HCL, nodeTypes: SPLITTABLE_NODE_TYPES.hcl },
      tf: { parser: HCL, nodeTypes: SPLITTABLE_NODE_TYPES.hcl },
      jinja2: { parser: Jinja2, nodeTypes: SPLITTABLE_NODE_TYPES.jinja2 },
      jinja: { parser: Jinja2, nodeTypes: SPLITTABLE_NODE_TYPES.jinja2 },
      j2: { parser: Jinja2, nodeTypes: SPLITTABLE_NODE_TYPES.jinja2 },
      kotlin: { parser: Kotlin, nodeTypes: SPLITTABLE_NODE_TYPES.kotlin },
      kt: { parser: Kotlin, nodeTypes: SPLITTABLE_NODE_TYPES.kotlin },
      kts: { parser: Kotlin, nodeTypes: SPLITTABLE_NODE_TYPES.kotlin },
      markdown: { parser: Markdown, nodeTypes: SPLITTABLE_NODE_TYPES.markdown },
      md: { parser: Markdown, nodeTypes: SPLITTABLE_NODE_TYPES.markdown },
      php: { parser: PHP, nodeTypes: SPLITTABLE_NODE_TYPES.php },
      prisma: { parser: Prisma, nodeTypes: SPLITTABLE_NODE_TYPES.prisma },
      scss: { parser: SCSS, nodeTypes: SPLITTABLE_NODE_TYPES.scss },
      sql: { parser: SQL, nodeTypes: SPLITTABLE_NODE_TYPES.sql },
      toml: { parser: TOML, nodeTypes: SPLITTABLE_NODE_TYPES.toml },
      vue: { parser: Vue, nodeTypes: SPLITTABLE_NODE_TYPES.vue },
      yaml: { parser: YAML, nodeTypes: SPLITTABLE_NODE_TYPES.yaml },
      yml: { parser: YAML, nodeTypes: SPLITTABLE_NODE_TYPES.yaml },
    };

    return langMap[language.toLowerCase()] || null;
  }

  private extractChunks(
    node: Parser.SyntaxNode,
    code: string,
    splittableTypes: string[],
    language: string,
    filePath?: string,
  ): CodeChunk[] {
    const chunks: CodeChunk[] = [];
    const codeLines = code.split("\n");

    const traverse = (currentNode: Parser.SyntaxNode) => {
      // Check if this node type should be split into a chunk
      if (splittableTypes.includes(currentNode.type)) {
        const startLine = currentNode.startPosition.row + 1;
        const endLine = currentNode.endPosition.row + 1;
        const nodeText = code.slice(
          currentNode.startIndex,
          currentNode.endIndex,
        );

        // Only create chunk if it has meaningful content
        if (nodeText.trim().length > 0) {
          chunks.push({
            content: nodeText,
            metadata: {
              startLine,
              endLine,
              language,
              filePath,
            },
          });
        }
      }

      // Continue traversing child nodes
      for (const child of currentNode.children) {
        traverse(child);
      }
    };

    traverse(node);

    // If no meaningful chunks found, create a single chunk with the entire code
    if (chunks.length === 0) {
      chunks.push({
        content: code,
        metadata: {
          startLine: 1,
          endLine: codeLines.length,
          language,
          filePath,
        },
      });
    }

    return chunks;
  }

  private async refineChunks(
    chunks: CodeChunk[],
    originalCode: string,
  ): Promise<CodeChunk[]> {
    const refinedChunks: CodeChunk[] = [];

    for (const chunk of chunks) {
      if (chunk.content.length <= this.chunkSize) {
        refinedChunks.push(chunk);
      } else {
        // Split large chunks using character-based splitting
        const subChunks = this.splitLargeChunk(chunk, originalCode);
        refinedChunks.push(...subChunks);
      }
    }

    return this.addOverlap(refinedChunks);
  }

  private splitLargeChunk(chunk: CodeChunk, originalCode: string): CodeChunk[] {
    const lines = chunk.content.split("\n");
    const subChunks: CodeChunk[] = [];
    let currentChunk = "";
    let currentStartLine = chunk.metadata.startLine;
    let currentLineCount = 0;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const lineWithNewline = i === lines.length - 1 ? line : line + "\n";

      if (
        currentChunk.length + lineWithNewline.length > this.chunkSize &&
        currentChunk.length > 0
      ) {
        // Create a sub-chunk
        subChunks.push({
          content: currentChunk.trim(),
          metadata: {
            startLine: currentStartLine,
            endLine: currentStartLine + currentLineCount - 1,
            language: chunk.metadata.language,
            filePath: chunk.metadata.filePath,
          },
        });

        currentChunk = lineWithNewline;
        currentStartLine = chunk.metadata.startLine + i;
        currentLineCount = 1;
      } else {
        currentChunk += lineWithNewline;
        currentLineCount++;
      }
    }

    // Add the last sub-chunk
    if (currentChunk.trim().length > 0) {
      subChunks.push({
        content: currentChunk.trim(),
        metadata: {
          startLine: currentStartLine,
          endLine: currentStartLine + currentLineCount - 1,
          language: chunk.metadata.language,
          filePath: chunk.metadata.filePath,
        },
      });
    }

    return subChunks;
  }

  private addOverlap(chunks: CodeChunk[]): CodeChunk[] {
    if (chunks.length <= 1 || this.chunkOverlap <= 0) {
      return chunks;
    }

    const overlappedChunks: CodeChunk[] = [];

    for (let i = 0; i < chunks.length; i++) {
      let content = chunks[i].content;
      const metadata = { ...chunks[i].metadata };

      // Add overlap from previous chunk
      if (i > 0 && this.chunkOverlap > 0) {
        const prevChunk = chunks[i - 1];
        const overlapText = prevChunk.content.slice(-this.chunkOverlap);
        content = overlapText + "\n" + content;
        metadata.startLine = Math.max(
          1,
          metadata.startLine - this.getLineCount(overlapText),
        );
      }

      overlappedChunks.push({
        content,
        metadata,
      });
    }

    return overlappedChunks;
  }

  private getLineCount(text: string): number {
    return text.split("\n").length;
  }

  /**
   * Check if AST splitting is supported for the given language
   */
  static isLanguageSupported(language: string): boolean {
    const supportedLanguages = [
      "javascript", "js",
      "typescript", "ts",
      "python", "py",
      "java",
      "cpp", "c++", "c",
      "go",
      "rust", "rs",
      "cs", "csharp",
      "scala",
      "tsx",
      "angular",
      "bash", "sh", "zsh",
      "css",
      "elixir", "ex", "exs",
      "hcl", "terraform", "tf",
      "html", "htm",
      "jinja2", "jinja", "j2",
      "json",
      "kotlin", "kt", "kts",
      "markdown", "md",
      "php",
      "prisma",
      "ruby", "rb",
      "scss",
      "sql",
      "toml",
      "vue",
      "yaml", "yml",
    ];
    return supportedLanguages.includes(language.toLowerCase());
  }
}
