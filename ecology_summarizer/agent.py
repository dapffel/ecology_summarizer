import instructor
import litellm

from .memory import VectorMemory
from .models import AgentConfig, StructuredSummary
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

    async def summarize_pdf(
        self, pdf_path: str, references: list[str] | None = None
    ) -> StructuredSummary:
        text = extract_text(pdf_path).strip()
        if not text:
            raise ValueError("PDF contains no extractable text")
        text = text[: self.config.max_input_chars]

        context = ""
        if references:
            memory = VectorMemory(
                model=self.config.embedding_model,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            await memory.add(references[: self.config.max_reference_docs])
            context_chunks = await memory.query(text[:500])
            context = "\n\n".join(context_chunks)

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
