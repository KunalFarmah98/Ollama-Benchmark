## Project: Ollama-bench

## Directory Structure
Run `tree /F /A` to get the full file list if needed.

## Key Files
- `src/ollama_bench/benchmark_impl.py` — Main implementation file for generating and validating project scaffolds
- `src/ollama_bench/benchmark.py` — Additional benchmarking logic and utilities
- `src/ollama_bench/__main__.py` — Entry point for command-line execution
- `src/ollama_bench/__init__.py` — Package initialization and module exports

## Rules
- ALWAYS run the `Glob` or `Bash` tool to find files before assuming their names
- NEVER guess file paths — use `Glob("**/*.py")` or `Bash("dir /s /b")` to locate files first
- When the user says "that file" or "the server", search for it using Glob before reading

## Skills
- No skills are currently available.

## Tools
- `Glob`: For searching files by pattern
- `Bash`: For executing shell commands
- `Read`: For reading file contents
- `Edit`: For modifying files
- `Write`: For creating new files
- `Task`: For complex, multi-step tasks
- `RAG`: For semantic search over indexed code

## Agents
- **Architecture Generator**: Handles phase 0, generating the project's overall architecture. Located in `src/ollama_bench/benchmark_impl.py`.
- **Scaffold Creator**: Manages phase 1, generating initial project files and validating the manifest. Located in `src/ollama_bench/benchmark_impl.py`.
- **Implementation Generator**: Conducts phase 2, batch-generating pending files and adjusting batch sizes as needed. Located in `src/ollama_bench/benchmark_impl.py`.
- **Validator**: Runs Gradle tests and smoke tests to ensure the generated project is correct. Located in `src/ollama_bench/benchmark_impl.py`.
- **Memory Monitor**: Tracks peak RAM and VRAM usage during model execution. Located in `src/ollama_bench/benchmark_impl.py`.

## Codebase Search — RAG Index

This project has a fully indexed RAG vector store via the `rag` MCP server.

### Tool Names
- `rag_search` — semantic search over the indexed codebase (use this first)
- `rag_health` — check if the RAG server is running and how many chunks are indexed

### Rules
- ALWAYS call `rag_search` before reading any source file
- Use natural language queries like "where is the embedding function defined"
- Only fall back to `Read` or `Glob` if `rag_search` returns 0 results

# Agent Rules
- Before any `edit` call, always `read` the target file in the same turn.
- `oldString` must be copied verbatim from read output, minimum 5 lines of context.
- On edit failure, re-read and retry once. On second failure, use `write` instead.

# SESSION RULES (apply for entire session):
- Active file = last file you read, listed, or touched. Use it as implicit target for follow-up commands.
- Never greet. Never announce plans. Execute first, output results only.
- On edit: always read the file first. Copy oldString verbatim with 5+ lines of context. On failure, re-read and retry once, then fall back to write.
- Never use apply_patch. Use edit or write only.