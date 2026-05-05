# Ecology Summarizer

Structured summaries of ecological research papers from PDFs. Provider-agnostic via [LiteLLM](https://github.com/BerriAI/litellm).

## Install

```bash
pip install .
```

## Setup

Add your API key to `.env`:

```
OPENAI_API_KEY=sk-...
```

Any LiteLLM-supported provider works — just set the relevant key (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, etc).

## Usage

```python
import asyncio
from ecology_summarizer import SummarizationAgent, AgentConfig

async def main():
    agent = SummarizationAgent()
    summary = await agent.summarize_pdf("paper.pdf")
    print(summary.title)
    print(summary.key_sentence)

asyncio.run(main())
```

Switch providers by changing the model string:

```python
config = AgentConfig(model="anthropic/claude-sonnet-4-6")
agent = SummarizationAgent(config)
```

Pass reference documents for context-aware summaries:

```python
summary = await agent.summarize_pdf(
    "paper.pdf",
    references=["Smith et al. 2023 found that..."]
)
```

## Output

`StructuredSummary` with fields: `title`, `study_context`, `methods`, `key_findings`, `ecological_implications`, `key_sentence`.
