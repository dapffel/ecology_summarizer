"""Quick test: extract SDM requirements from a paper and optionally evaluate."""

import asyncio
import json
import sys

from lit_review import AgentConfig, SDMExtractionAgent


async def main():
    if len(sys.argv) < 2:
        print("Usage: python try_extract.py <path-to-pdf> [model]")
        print("Example: python try_extract.py paper.pdf")
        print("         python try_extract.py paper.pdf anthropic/claude-sonnet-4-6")
        sys.exit(1)

    pdf_path = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "gpt-4o"

    config = AgentConfig(model=model)
    agent = SDMExtractionAgent(config)

    print(f"Extracting from: {pdf_path}")
    print(f"Model: {model}\n")

    result = await agent.extract_from_pdf(pdf_path)

    print("=== Extraction ===")
    print(json.dumps(result.model_dump(exclude_none=True), indent=2))

    print("\n=== Evaluation ===")
    evaluation = await agent.evaluate(result, pdf_path)
    print(json.dumps(evaluation.model_dump(), indent=2))

    print(f"\nVerified: {evaluation.num_verified}")
    print(f"Inaccurate: {evaluation.num_inaccurate}")
    print(f"Unverifiable: {evaluation.num_unverifiable}")


asyncio.run(main())
