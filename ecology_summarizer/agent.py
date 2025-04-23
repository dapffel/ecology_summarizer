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
from pydantic import BaseModel, ValidationError, validator
from contextlib import asynccontextmanager
import re
from transformers import GPT2TokenizerFast
from collections import deque
from asyncio import Lock
from openai import AsyncOpenAI, OpenAIError, RateLimitError, APIError
import time

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
    max_reference_docs: int = 10
    model_price: Dict[ModelType, Tuple[float, float]] = field(
        default_factory=lambda: {
            ModelType.GPT_4: (0.03, 0.06),
            ModelType.GPT_3_5: (0.0015, 0.002)
        }
    )

    def __post_init__(self):
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= self.temperature <= 1:
            raise ValueError("temperature must be between 0 and 1")
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        if self.max_reference_docs <= 0:
            raise ValueError("max_reference_docs must be positive")
        
        # Validate model prices
        for model, (input_rate, output_rate) in self.model_price.items():
            if input_rate < 0 or output_rate < 0:
                raise ValueError(f"Model prices for {model} must be non-negative")

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
    MAX_TOKENS_PER_CHUNK = 8191  # OpenAI's embedding model limit

    def __init__(self, dimension: int = 1536, max_texts: int = 1000, api_key: Optional[str] = None):
        self.index = faiss.IndexFlatL2(dimension)
        self.texts: deque = deque(maxlen=max_texts)
        self.dimension = dimension
        self.embeddings = np.zeros((0, dimension), dtype=np.float32)
        self._lock = Lock()
        self._embedding_client = AsyncOpenAI(api_key=api_key)


    async def _embed(self, text: str) -> np.ndarray:
        response = await self._embedding_client.embeddings.create(
            model=ModelType.EMBEDDING.value,
            input=text
        )
        logger.info(f"Embedded text of length {len(text)} tokens")
        return np.array(response.data[0].embedding, dtype=np.float32)

    async def add(self, texts: List[str], splitter: Optional["SmartTextSplitter"] = None):
        async with self._lock:
            new_embeddings = []
            total_chunks = 0
            skipped_chunks = 0

            for text in texts:
                # Use splitter if available
                chunks = [text]
                if splitter:
                    chunks = [chunk.text for chunk in await splitter.split_text(text)]

                for chunk in chunks:
                    total_chunks += 1
                    try:
                        # Use proper token counting if splitter is available
                        token_count = await splitter.count_tokens(chunk) if splitter else len(chunk.split())
                        
                        if token_count > self.MAX_TOKENS_PER_CHUNK:
                            # Split the chunk into smaller parts
                            if not splitter:
                                # Create a temporary splitter if none provided
                                temp_splitter = SmartTextSplitter(max_tokens=self.MAX_TOKENS_PER_CHUNK)
                                sub_chunks = [chunk.text for chunk in await temp_splitter.split_text(chunk)]
                                temp_splitter.cleanup()
                            else:
                                sub_chunks = [chunk.text for chunk in await splitter.split_text(chunk)]
                            
                            logger.info(f"Splitting large chunk ({token_count} tokens) into {len(sub_chunks)} sub-chunks")
                            
                            # Process each sub-chunk
                            for sub_chunk in sub_chunks:
                                try:
                                    emb = await self._embed(sub_chunk)
                                    new_embeddings.append(emb)
                                    self.texts.append(sub_chunk)
                                except Exception as e:
                                    logger.warning(f"Failed to embed sub-chunk: {e}")
                                    skipped_chunks += 1
                            continue

                        emb = await self._embed(chunk)
                        new_embeddings.append(emb)
                        self.texts.append(chunk)
                    except OpenAIError as e:
                        logger.warning(f"OpenAI error for chunk: {e}")
                        skipped_chunks += 1
                    except RateLimitError as e:
                        logger.error(f"Rate limit exceeded: {e}")
                        raise
                    except APIError as e:
                        logger.error(f"API error: {e}")
                        raise
                    except Exception as e:
                        logger.warning(f"Unexpected error embedding chunk: {e}")
                        skipped_chunks += 1

            if not new_embeddings:
                logger.warning("No embeddings added. All chunks failed or were skipped.")
                return

            logger.info(f"Added {len(new_embeddings)} chunks successfully. "
                       f"Skipped {skipped_chunks} out of {total_chunks} total chunks.")

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
    def __init__(self, max_tokens: int = 250):
        self.max_tokens = max_tokens
        self.tokenizer = None
        self._tokenizer_lock = asyncio.Lock()
    
    async def _ensure_tokenizer(self):
        if self.tokenizer is None:
            try:
                async with self._tokenizer_lock:
                    if self.tokenizer is None:  # Double-check pattern
                        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                        logger.info("Tokenizer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize tokenizer: {e}")
                raise
    
    def cleanup(self):
        if self.tokenizer:
            try:
                del self.tokenizer
                self.tokenizer = None
                logger.info("Tokenizer cleaned up successfully")
            except Exception as e:
                logger.error(f"Error cleaning up tokenizer: {e}")

    async def count_tokens(self, text: str) -> int:
        await self._ensure_tokenizer()
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Fallback to word count if tokenizer fails
            return len(text.split())

    async def split_text(self, text: str) -> List[TextChunk]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[TextChunk] = []
        current_chunk: List[str] = []
        current_count = 0

        for para in paragraphs:
            tokens = await self.count_tokens(para)
            if tokens > self.max_tokens:
                for sent in re.split(r'(?<=[.!?])\s+', para):
                    stoks = await self.count_tokens(sent)
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
        self.text_splitter = SmartTextSplitter(max_tokens=250)
        self._setup()
        self._chat_client = None
        self._rate_limit_semaphore = asyncio.Semaphore(1)
        self._last_api_call = 0
        self._min_call_interval = 10.0
        self._token_budget = 5000
        self._tokens_used = 0
        self._token_reset_time = time.time()
        self._token_reset_interval = 60

    def _setup(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment")
        self._chat_client = AsyncOpenAI(api_key=api_key)
        self.memory = VectorMemory(dimension=self.config.embedding_dimension, api_key=api_key)

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
        try:
            if hasattr(self, 'memory'):
                del self.memory.index
            if hasattr(self, 'text_splitter'):
                self.text_splitter.cleanup()
            if self._chat_client:
                await self._chat_client.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            # Don't raise during cleanup to avoid masking original error

    async def process_large_text(self, text: str) -> AsyncIterator[TextChunk]:
        chunks = await self.text_splitter.split_text(text)
        for chunk in chunks:
            yield chunk

    async def _parse_summary(self, summary_text: str) -> StructuredSummary:
        try:
            return StructuredSummary.from_text(summary_text)
        except ValidationError as e:
            logger.error(f"Failed to parse summary: {e}")
            raise

    def _estimate_tokens(self, text: str) -> int:
        """More sophisticated token estimation"""
        # Count words and characters
        words = len(text.split())
        chars = len(text)
        
        # Count special tokens (headers, lists, etc.)
        special_tokens = sum(1 for _ in re.finditer(r'[-*#]', text))
        
        # More conservative estimation
        word_tokens = words * 1.3  # Increased from 1.5 to account for longer words
        char_tokens = chars / 3.5  # Increased from 4 to be more conservative
        special_token_count = special_tokens * 2  # Account for special formatting
        
        # Take the maximum of the estimates and add special tokens
        return int(max(word_tokens, char_tokens) + special_token_count)

    def _update_token_budget(self, tokens: int):
        current_time = time.time()
        if current_time - self._token_reset_time >= self._token_reset_interval:
            self._tokens_used = 0
            self._token_reset_time = current_time
        
        # Add 20% buffer to token count
        tokens_with_buffer = int(tokens * 1.2)
        
        self._tokens_used += tokens_with_buffer
        if self._tokens_used >= self._token_budget:
            wait_time = self._token_reset_interval - (current_time - self._token_reset_time)
            if wait_time > 0:
                logger.info(f"Token budget exceeded, waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
                self._tokens_used = 0
                self._token_reset_time = time.time()

    async def _rate_limited_api_call(self, coro, estimated_tokens: int = 0):
        async with self._rate_limit_semaphore:
            self._update_token_budget(estimated_tokens)
            
            current_time = time.time()
            time_since_last_call = current_time - self._last_api_call
            if time_since_last_call < self._min_call_interval:
                await asyncio.sleep(self._min_call_interval - time_since_last_call)
            
            try:
                result = await coro
                self._last_api_call = time.time()
                return result
            except RateLimitError as e:
                logger.warning(f"Rate limit hit, waiting before retry: {e}")
                await asyncio.sleep(30)  # Increased from 20 to 30 seconds
                raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _get_completion(self, messages: List[Dict[str, str]]):
        try:
            if not self._chat_client:
                self._setup()
                
            # More accurate token estimation with buffer
            estimated_tokens = sum(self._estimate_tokens(msg["content"]) for msg in messages)
            
            # Add buffer for system message and response
            estimated_tokens = int(estimated_tokens * 1.2)
            
            async def make_request():
                return await self._chat_client.chat.completions.create(
                    model=self.config.model.value,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
            
            response = await self._rate_limited_api_call(make_request(), estimated_tokens)
            content = response.choices[0].message.content
            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            cost = ((prompt_tokens / 1000) * self.config.model_price[self.config.model][0]
                    + (completion_tokens / 1000) * self.config.model_price[self.config.model][1])
            return CompletionResult(content, cost)
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            raise
        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in _get_completion: {e}")
            raise

    async def _embed(self, text: str) -> np.ndarray:
        # More accurate token estimation with buffer
        estimated_tokens = int(self._estimate_tokens(text) * 1.2)
        
        async def make_request():
            response = await self._embedding_client.embeddings.create(
                model=ModelType.EMBEDDING.value,
                input=text
            )
            logger.info(f"Embedded text of length {len(text)} tokens")
            return np.array(response.data[0].embedding, dtype=np.float32)
        
        return await self._rate_limited_api_call(make_request(), estimated_tokens)

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
