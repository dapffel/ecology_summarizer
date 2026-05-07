import instructor
import litellm
from dotenv import load_dotenv

from .memory import VectorMemory
from .models import AgentConfig, SDMRequirements
from .pdf import extract_text

load_dotenv()

SYSTEM_PROMPT = (
    "You are an expert in species distribution modeling (SDM) and ecological niche modeling. "
    "Your task is to extract structured technical requirements from the provided research paper. "
    "Focus specifically on the methodological details needed to reproduce the SDM analysis: "
    "species identity, occurrence data characteristics, environmental predictors, "
    "modeling algorithm and settings, evaluation strategy, and key results.\n\n"
    "Extraction guidelines:\n"
    "- Be precise: use exact numbers, variable names, and software versions from the paper.\n"
    "- If the paper reports multiple models, focus on the best-performing or recommended model.\n"
    "- If a detail is not mentioned in the paper, return null for that field rather than guessing.\n"
    "- For environmental variables, distinguish between candidate variables and those retained "
    "after selection.\n"
    "- For ensemble models, list all component algorithms.\n"
    "- Include units where relevant (e.g., spatial resolution in meters or arc-seconds)."
)

PAPER_PREFIX = "SDM Paper:\n\n"
CONTEXT_PREFIX = "Reference SDM methodology context:\n"
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


class SDMExtractionAgent:
    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self.client = instructor.from_litellm(litellm.acompletion)

    async def extract_from_pdf(
        self, pdf_path: str, references: list[str] | None = None
    ) -> SDMRequirements:
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
            response_model=SDMRequirements,
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
