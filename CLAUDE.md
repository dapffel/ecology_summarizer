# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
mypy lit_review/                    # type check
```

## Environment

Set provider API keys in `.env` at the project root (loaded via python-dotenv). At minimum `OPENAI_API_KEY` for embeddings. Any LiteLLM-supported provider works for completions.

## Architecture

Single agent that extracts species distribution modeling (SDM) requirements from research PDFs into a validated Pydantic model. The extracted data is designed to drive virtual species experiments.

- **`agent.py`** — `SDMExtractionAgent` orchestrates the pipeline: extract PDF text, optionally retrieve context from reference SDM papers via vector memory, call LLM via Instructor+LiteLLM to get an `SDMRequirements` model directly (no regex parsing).
- **`models.py`** — Pydantic v2 `AgentConfig`, nested `SDMRequirements` with typed fields (lists, ints, floats), per-model `SDMModelSpec` with numeric `PerformanceMetric`, and `ExtractionEval`. Field descriptions guide the LLM's structured extraction.
- **`prompts.py`** — All prompt text for extraction and evaluation, separated from pipeline logic.
- **`memory.py`** — `VectorMemory` wraps FAISS for in-memory vector search. Used to provide context from reference SDM papers during extraction. Embeds via `litellm.aembedding()`, splits text via `langchain-text-splitters`.
- **`pdf.py`** — PDF text extraction via PyMuPDF.

Provider-agnostic: pass any LiteLLM model string (e.g. `"gpt-4"`, `"anthropic/claude-sonnet-4-6"`, `"gemini/gemini-pro"`) in `AgentConfig.model`.
