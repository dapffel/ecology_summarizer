import instructor
import litellm

from .models import AgentConfig, StructuredSummary
from .memory import VectorMemory
from .pdf import extract_text

SYSTEM_PROMPT = (
    "You are an expert ecological researcher and scientific writer. "
    "Summarize the following ecological research paper accurately and concisely, "
    "using scientific language."
)


class SummarizationAgent:
    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self.client = instructor.from_litellm(litellm.acompletion)
        self.memory = VectorMemory(
            model=self.config.embedding_model,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    async def summarize_pdf(
        self, pdf_path: str, references: list[str] | None = None
    ) -> StructuredSummary:
        text = extract_text(pdf_path)

        if references:
            await self.memory.add(references[: self.config.max_reference_docs])

        context_chunks = await self.memory.query(text[:500])
        context = "\n\n".join(context_chunks) if context_chunks else ""

        user_content = f"Paper:\n\n{text}"
        if context:
            user_content = f"Relevant background:\n{context}\n\n{user_content}"

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
