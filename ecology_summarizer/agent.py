import os
import faiss
import numpy as np
import openai
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, AsyncIterator, Tuple
from dotenv import load_dotenv
from dataclasses import dataclass, field
from enum import Enum
from tenacity import retry, stop_after_attempt, wait_exponential
from PyPDF2 import PdfReader
from pydantic import BaseModel, ValidationError
from contextlib import asynccontextmanager
import re
from transformers import GPT2TokenizerFast
from collections import deque
from asyncio import Lock

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelType(Enum):
    GPT_3_5 = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    EMBEDDING = "text-embedding-ada-002"

@dataclass
class AgentConfig:
    model: ModelType = ModelType.GPT_4
    temperature: float = 0.5
    max_tokens: int = 1000
    embedding_dimension: int = 1536
    structured_summary_prompt: str = (
        "You are an expert ecological researcher and scientific writer.\n"
        "Summarize the following ecological research paper using this format:\n\n"
        "- **Title**:\n"
        "- **Study Context**:\n"
        "- **Methods**:\n"
        "- **Key Findings**:\n"
        "- **Ecological Implications**:\n\n"
        "Then provide a one-sentence summary suitable for inclusion in a research proposal under:\n\n"
        "- **Key Sentence for Research Proposal**:\n\n"
        "Be accurate, concise, and use scientific language."
    )
    max_reference_docs: int = 10  # Max number of references to embed
    model_price: Dict[ModelType, Tuple[float, float]] = field(
        default_factory=lambda: {
            ModelType.GPT_4: (0.03, 0.06),      # input, output $ per 1K tokens
            ModelType.GPT_3_5: (0.0015, 0.002)   # input, output $ per 1K tokens
        }
    )

class StructuredSummary(BaseModel):
    title: str
    study_context: str
    methods: str
    key_findings: str
    ecological_implications: str
    key_sentence_for_research_proposal: str

    @validator('*')
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

    @classmethod
    def from_text(cls, text: str) -> "StructuredSummary":
        # Helper to extract sections
        def extract_section(header: str) -> str:
            pattern = rf"- \*\*{header}\*\*: ?(.+?)(?=\n- \*\*|$)"
            match = re.search(pattern, text, re.DOTALL)
            return match.group(1).strip() if match else ""

        return cls(
            title=extract_section("Title"),
            study_context=extract_section("Study Context"),
            methods=extract_section("Methods"),
            key_findings=extract_section("Key Findings"),
            ecological_implications=extract_section("Ecological Implications"),
            key_sentence_for_research_proposal=extract_section("Key Sentence for Research Proposal")
        )

@dataclass
class TextChunk:
    text: str
    tokens: int
    metadata: Dict = field(default_factory=dict)

class VectorMemory:
    def __init__(self, dimension: int = 1536, max_texts: int = 1000):
        self.index = faiss.IndexFlatL2(dimension)
        self.texts: deque = deque(maxlen=max_texts)
        self.dimension = dimension
        self.embeddings = np.zeros((0, dimension), dtype=np.float32)
        self._lock = Lock()

    async def _embed(self, text: str) -> np.ndarray:
        response = await openai.Embedding.acreate(
            model=ModelType.EMBEDDING.value,
            input=text
        )
        logger.info(f"Embedded text of length {len(text)} tokens")
        return np.array(response["data"][0]["embedding"], dtype=np.float32)

    async def add(self, texts: List[str]):
        async with self._lock:
            new_embeddings = []
            for text in texts:
                emb = await self._embed(text)
                new_embeddings.append(emb)
                self.texts.append(text)

            new_array = np.vstack(new_embeddings)
            self.embeddings = new_array if self.embeddings.size == 0 else np.vstack([self.embeddings, new_array])
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(self.embeddings)
            logger.info(f"Vector memory now contains {len(self.texts)} documents")

    async def query(self, prompt: str, k: int = 3) -> List[str]:
        if not self.texts:
            return []
        emb = await self._embed(prompt)
        D, I = self.index.search(np.array([emb]), k)
        logger.info(f"Queried vector memory, returning top {k} results")
        return [self.texts[i] for i in I[0]]

async def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    logger.info(f"Extracted {len(text.split())} words from PDF: {pdf_path}")
    return text

class SmartTextSplitter:
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.tokenizer = None
    
    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    def cleanup(self):
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def split_text(self, text: str) -> List[TextChunk]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[TextChunk] = []
        current_chunk: List[str] = []
        current_count = 0

        for para in paragraphs:
            tokens = self.count_tokens(para)
            if tokens > self.max_tokens:
                for sent in re.split(r'(?<=[.!?])\s+', para):
                    stoks = self.count_tokens(sent)
                    if current_count + stoks <= self.max_tokens:
                        current_chunk.append(sent)
                        current_count += stoks
                    else:
                        chunks.append(TextChunk(text=' '.join(current_chunk), tokens=current_count))
                        current_chunk = [sent]
                        current_count = stoks
            else:
                if current_count + tokens <= self.max_tokens:
                    current_chunk.append(para)
                    current_count += tokens
                else:
                    chunks.append(TextChunk(text='\n\n'.join(current_chunk), tokens=current_count))
                    current_chunk = [para]
                    current_count = tokens

        if current_chunk:
            chunks.append(TextChunk(text='\n\n'.join(current_chunk), tokens=current_count))
        return chunks

class SummarizationAgent:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.text_splitter = SmartTextSplitter(max_tokens=4000)
        self._setup()

    def _setup(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment")
        self.memory = VectorMemory(dimension=self.config.embedding_dimension)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def summarize_pdf(self, pdf_path: str, reference_texts: List[str], max_cost_usd: float = 1.0) -> StructuredSummary:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if os.path.getsize(pdf_path) > 100 * 1024 * 1024:
            raise ValueError("PDF file too large")

        if reference_texts:
            await self.memory.add(reference_texts[:self.config.max_reference_docs])

        current_cost = 0.0
        paper_text = await extract_text_from_pdf(pdf_path)
        summaries: List[str] = []

        async for chunk in self.process_large_text(paper_text):
            est = self._estimate_cost(chunk.tokens)
            if current_cost + est > max_cost_usd:
                logger.warning("Cost limit reached, stopping processing")
                break
            context = await self.memory.query(chunk.text)
            messages = build_summarization_prompt(chunk.text, context, self.config)
            response = await self._get_completion(messages)
            current_cost += response.cost
            summaries.append(response.content)

        if not summaries:
            raise ValueError("No text chunks were processed")
        return await self._merge_summaries(summaries)

    async def _merge_summaries(self, summaries: List[str]) -> StructuredSummary:
        merge_prompt = (
            f"Merge these {len(summaries)} section summaries into a single coherent summary." +
            " Maintain the same structured format but combine overlapping information and resolve any contradictions:\n\n" +
            "\n\n".join(summaries)
        )
        messages = [
            {"role": "system", "content": self.config.structured_summary_prompt},
            {"role": "user", "content": merge_prompt}
        ]
        response = await self._get_completion(messages)
        return await self._parse_summary(response.content)

    def _estimate_cost(self, num_tokens: int) -> float:
        in_rate, out_rate = self.config.model_price[self.config.model]
        return (num_tokens / 1000) * (in_rate + out_rate)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'memory'):
            del self.memory.index
        if hasattr(self, 'text_splitter'):
            self.text_splitter.cleanup()

    async def process_large_text(self, text: str) -> AsyncIterator[TextChunk]:
        for chunk in self.text_splitter.split_text(text):
            yield chunk

    async def _parse_summary(self, summary_text: str) -> StructuredSummary:
        try:
            return StructuredSummary.from_text(summary_text)
        except ValidationError as e:
            logger.error(f"Failed to parse summary: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _get_completion(self, messages: List[Dict[str, str]]):
        response = await openai.ChatCompletion.acreate(
            model=self.config.model.value,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=False
        )
        content = response.choices[0].message.content
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        cost = ((prompt_tokens / 1000) * self.config.model_price[self.config.model][0]
                + (completion_tokens / 1000) * self.config.model_price[self.config.model][1])
        return CompletionResult(content, cost)

def build_summarization_prompt(paper_text: str, context_chunks: List[str], config: AgentConfig) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": config.structured_summary_prompt}]
    if context_chunks:
        messages.append({
            "role": "user",
            "content": f"Here is helpful background:\n{'\n\n'.join(context_chunks)}"
        })
    messages.append({
        "role": "user",
        "content": f"Summarize this paper:\n\n{paper_text}"
    })
    return messages

@dataclass
class CompletionResult:
    content: str
    cost: float
