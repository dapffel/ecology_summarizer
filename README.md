# lit-review

Structured SDM requirements extraction from research PDFs. Provider-agnostic via [LiteLLM](https://github.com/BerriAI/litellm).

Extracts species distribution modeling methodology from papers into a structured format suitable for driving virtual species experiments.

## Install

```bash
pip install .
```

For development:

```bash
pip install -e ".[dev]"
pytest
black .
isort .
mypy lit_review/
```

## Setup

Add your API key to `.env` at the project root. It is loaded automatically via `python-dotenv`.

```
OPENAI_API_KEY=sk-...
```

Any LiteLLM-supported provider works — just set the relevant key (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, etc).

## Usage

```python
import asyncio
from lit_review import SDMExtractionAgent, AgentConfig

async def main():
    agent = SDMExtractionAgent()
    result = await agent.extract_from_pdf("paper.pdf")
    print(result.study.title)
    print(result.study.species)              # ['Bufo marinus']
    print(result.predictors.variables)       # ['BIO1', 'BIO12', ...]
    for m in result.models:
        print(f"{m.algorithm}: {m.performance}")

asyncio.run(main())
```

Switch providers by changing the model string:

```python
config = AgentConfig(model="anthropic/claude-sonnet-4-6")
agent = SDMExtractionAgent(config)
```

Limit prompt size or customize embeddings through `AgentConfig`:

```python
config = AgentConfig(
    model="anthropic/claude-sonnet-4-6",
    embedding_model="text-embedding-ada-002",
    max_input_chars=100_000,
)
agent = SDMExtractionAgent(config)
```

Pass reference documents for context-aware extraction:

```python
result = await agent.extract_from_pdf(
    "paper.pdf",
    references=["Smith et al. 2023 used MaxEnt with spatial block CV..."]
)
```

References are used only for the current `extract_from_pdf()` call; memory is not persisted across extractions.

Run a verification pass to cross-check extracted values against the source PDF:

```python
evaluation = await agent.evaluate(result, "paper.pdf")
print(evaluation.num_verified)
print(evaluation.overall_assessment)
```

## Output

`SDMRequirements` is grouped into methodology sections with machine-readable typed fields:

- `study` — `title: str`, `species: list[str]`, `geographic_extent: str`
- `occurrence` — `occurrence_type: str`, `total_presences: int`, `total_absences: int`, `data_source: str`
- `predictors` — `variables: list[str]`, `data_source: str`, `spatial_resolution: str`
- `models` — `list[SDMModelSpec]`, one per algorithm/variant tested. Each has `algorithm: str`, `software: str`, `hyperparameters: str`, `performance: list[PerformanceMetric]`, `is_best: bool`
- `evaluation` — `cv_strategy: str`, `metrics_used: list[str]`, `threshold_method: str`
- `results` — `key_predictors: list[str]`, `projected_scenarios: list[ProjectedScenario]`

Performance metrics are numeric: `PerformanceMetric(metric="AUC", value=0.92, std=0.03)`. Each section has an `evidence` field for provenance. Missing details default to `None` or `[]`.
