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
    print(result.model.algorithm.value)
    print(result.evaluation.metrics.value)

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

`SDMRequirements` is grouped into methodology sections:

- `study` — title, species, geographic extent
- `occurrence` — occurrence type, sample size, occurrence source
- `predictors` — environmental variables, data source, spatial resolution
- `model` — algorithm, software, hyperparameters
- `evaluation` — metrics, cross-validation, thresholding, performance values
- `results` — key predictors and predicted distribution

Most extracted methodology items are `ExtractedField` objects with `value`, `evidence`, `section`, and `page`. Use `value` for downstream virtual species workflows and `evidence` to audit where the value came from. Missing details are represented as `value=None` since papers vary in what they report.
