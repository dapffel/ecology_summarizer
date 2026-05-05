# Ecology Summarizer

A Python package for generating structured summaries of ecological research papers. Provider-agnostic — works with OpenAI, Anthropic, Cohere, Gemini, or any LiteLLM-supported model.

## Installation

```bash
git clone https://github.com/dapffel/ecology_summarizer.git
cd ecology_summarizer
pip install .
```

## Setup

Set your API key in a `.env` file:

```
OPENAI_API_KEY=sk-...
```

Or for other providers:
```
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

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

### Custom provider

```python
config = AgentConfig(
    model="anthropic/claude-sonnet-4-6",
    embedding_model="text-embedding-ada-002",
    temperature=0.3,
)
agent = SummarizationAgent(config)
```

### With reference documents

```python
references = [
    "Smith et al. 2023 found that forest fragmentation...",
    "Johnson 2022 showed biodiversity metrics...",
]
summary = await agent.summarize_pdf("paper.pdf", references=references)
```

## Output

Returns a `StructuredSummary` with fields:
- `title`
- `study_context`
- `methods`
- `key_findings`
- `ecological_implications`
- `key_sentence` — one-sentence summary for research proposals
