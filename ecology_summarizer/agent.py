import instructor
import litellm
from dotenv import load_dotenv

from .memory import VectorMemory
from .models import AgentConfig, StructuredSummary
from .pdf import extract_text

load_dotenv()

SYSTEM_PROMPT = (
    "You are an expert ecological researcher and scientific writer. "
    "Summarize the following ecological research paper accurately and concisely, "
    "using scientific language."
)

PAPER_PREFIX = "Paper:\n\n"
CONTEXT_PREFIX = "Relevant background:\n"
QUERY_CHARS = 500
MIN_PARAGRAPH_CHARS = 80


def _retrieval_query(text: str) -> str:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n")]
    for paragraph in paragraphs:
        if len(paragraph) >= MIN_PARAGRAPH_CHARS:
            return paragraph[:QUERY_CHARS]
    return text[:QUERY_CHARS]


def _build_user_content(text: str, context: str, max_chars: int) -> str:
    if not context:
        return f"{PAPER_PREFIX}{text}"[:max_chars]

    prefix = f"{CONTEXT_PREFIX}{context}\n\n{PAPER_PREFIX}"
    available = max_chars - len(prefix)
    if available <= 0:
        context_budget = max_chars - len(CONTEXT_PREFIX) - len(f"\n\n{PAPER_PREFIX}")
        context = context[: max(0, context_budget)]
        return f"{CONTEXT_PREFIX}{context}\n\n{PAPER_PREFIX}"[:max_chars]

    return f"{prefix}{text[:available]}"


class SummarizationAgent:
    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self.client = instructor.from_litellm(litellm.acompletion)

    async def summarize_pdf(
        self, pdf_path: str, references: list[str] | None = None
    ) -> StructuredSummary:
        text = extract_text(pdf_path).strip()
        if not text:
            raise ValueError("PDF contains no extractable text")

        context = ""
        if references:
            memory = VectorMemory(
                model=self.config.embedding_model,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            await memory.add(references[: self.config.max_reference_docs])
            context_chunks = await memory.query(_retrieval_query(text))
            context = "\n\n".join(context_chunks)

        user_content = _build_user_content(text, context, self.config.max_input_chars)

        return await self.client.create(
            model=self.config.model,
            response_model=StructuredSummary,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=self.config.temperature,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass
