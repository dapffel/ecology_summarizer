# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Build & Install

```bash
pip install .                # install package
pip install -e ".[dev]"      # editable install with dev tools (pytest, black, isort, mypy)
```

## Development Commands

```bash
pytest                              # run all tests
pytest tests/test_foo.py::test_bar  # run a single test
black .                             # format
isort .                             # sort imports
mypy ecology_summarizer/            # type check
```

## Environment

Set provider API keys in `.env` at the project root (loaded via python-dotenv). At minimum `OPENAI_API_KEY` for embeddings. Any LiteLLM-supported provider works for completions.

## Architecture

Single agent that summarizes ecological research PDFs into a validated Pydantic model.

- **`agent.py`** — `SummarizationAgent` orchestrates the pipeline: extract PDF text, optionally retrieve context from vector memory, call LLM via Instructor+LiteLLM to get a `StructuredSummary` directly (no regex parsing).
- **`models.py`** — Pydantic v2 `AgentConfig` and `StructuredSummary`. Field descriptions guide the LLM's structured output.
- **`memory.py`** — `VectorMemory` wraps FAISS for in-memory vector search. Embeds via `litellm.aembedding()`, splits text via `langchain-text-splitters`.
- **`pdf.py`** — PDF text extraction via PyMuPDF.

Provider-agnostic: pass any LiteLLM model string (e.g. `"gpt-4"`, `"anthropic/Codex-sonnet-4-6"`, `"gemini/gemini-pro"`) in `AgentConfig.model`.
