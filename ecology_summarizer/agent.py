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
    structured_summary_prompt: str = """You are an expert ecological researcher and scientific writer.\nSummarize the following ecological research paper using this format:\n\n- **Title**:\n- **Study Context**:\n- **Methods**:\n- **Key Findings**:\n- **Ecological Implications**:\n\nThen provide a one-sentence summary suitable for inclusion in a research proposal under:\n\n- **Key Sentence for Research Proposal**:\n\nBe accurate, concise, and use scientific language."""
    max_reference_docs: int = 10  # Max number of references to embed

class StructuredSummary(BaseModel):
    title: str
    study_context: str
    methods: str
    key_findings: str
    ecological_implications: str
    key_sentence_for_research_proposal: str

@dataclass
class TextChunk:
    text: str
    tokens: int
    metadata: Dict = field(default_factory=dict)

class VectorMemory:
    def __init__(self, dimension: int = 1536, max_texts: int = 1000):
        self.index = faiss.IndexFlatL2(dimension)
        self.texts: deque = deque(maxlen=max_texts)  # Using deque for automatic FIFO
        self.dimension = dimension
        self.embeddings = np.zeros((0, dimension), dtype=np.float32)
    
    async def _embed(self, text: str) -> np.ndarray:
        response = await openai.Embedding.acreate(
            model=ModelType.EMBEDDING.value,
            input=text
        )
        logger.info(f"Embedded text of length {len(text)} tokens")
        return np.array(response["data"][0]["embedding"], dtype=np.float32)

    async def add(self, texts: List[str]):
        new_embeddings = []
        for text in texts:
            emb = await self._embed(text)
            new_embeddings.append(emb)
            self.texts.append(text)
        
        # Update FAISS index with new embeddings
        new_embeddings_array = np.vstack(new_embeddings)
        if self.embeddings.shape[0] == 0:
            self.embeddings = new_embeddings_array
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings_array])
        
        # Rebuild index if needed
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.embeddings)
        
        logger.info(f"Vector memory now contains {len(self.texts)} documents")

    async def query(self, prompt: str, k=3) -> List[str]:
        if not self.texts:
            return []
        emb = await self._embed(prompt)
        D, I = self.index.search(np.array([emb]), k)
        logger.info(f"Queried vector memory, returning top {k} results")
        return [self.texts[i] for i in I[0]]

async def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    logger.info(f"Extracted {len(text.split())} words from PDF: {pdf_path}")
    return text

def build_summarization_prompt(paper_text: str, context_chunks: List[str], config: AgentConfig) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": config.structured_summary_prompt}]

    if context_chunks:
        retrieved_texts = "\n\n".join(context_chunks)
        messages.append({
            "role": "user",
            "content": f"Here is helpful background:\n{retrieved_texts}"
        })

    messages.append({
        "role": "user",
        "content": f"Summarize this paper:\n\n{paper_text}"
    })

    logger.info(f"Built summarization prompt with {len(messages)} message blocks")
    return messages

async def extract_texts_from_pdf_files(pdf_paths: List[str]) -> List[str]:
    texts = []
    for path in pdf_paths:
        try:
            text = await extract_text_from_pdf(path)
            if text:
                texts.append(text)
        except Exception as e:
            logger.warning(f"Failed to extract text from {path}: {e}")
    return texts

class SmartTextSplitter:
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def split_text(self, text: str) -> List[TextChunk]:
        # Split into paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks: List[TextChunk] = []
        current_chunk = []
        current_token_count = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = self.count_tokens(paragraph)
            
            # If single paragraph exceeds max tokens, split by sentences
            if paragraph_tokens > self.max_tokens:
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                for sentence in sentences:
                    sentence_tokens = self.count_tokens(sentence)
                    
                    if current_token_count + sentence_tokens <= self.max_tokens:
                        current_chunk.append(sentence)
                        current_token_count += sentence_tokens
                    else:
                        if current_chunk:
                            chunk_text = ' '.join(current_chunk)
                            chunks.append(TextChunk(
                                text=chunk_text,
                                tokens=current_token_count
                            ))
                        current_chunk = [sentence]
                        current_token_count = sentence_tokens
            
            # Normal paragraph processing
            elif current_token_count + paragraph_tokens <= self.max_tokens:
                current_chunk.append(paragraph)
                current_token_count += paragraph_tokens
            else:
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(TextChunk(
                        text=chunk_text,
                        tokens=current_token_count
                    ))
                current_chunk = [paragraph]
                current_token_count = paragraph_tokens
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                tokens=current_token_count
            ))
        
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
    async def summarize_pdf(self, pdf_path: str, reference_texts: List[str], 
                           max_cost_usd: float = 1.0) -> StructuredSummary:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        # Add file size check
        if os.path.getsize(pdf_path) > 100 * 1024 * 1024:  # 100MB limit
            raise ValueError("PDF file too large")

        # Track costs
        current_cost = 0.0
        
        # Extract and process text in chunks
        paper_text = await extract_text_from_pdf(pdf_path)
        chunks = []
        summaries = []
        
        async for chunk in self.process_large_text(paper_text):
            # Estimate cost before processing
            estimated_cost = self._estimate_cost(chunk.tokens)
            if current_cost + estimated_cost > max_cost_usd:
                logger.warning("Cost limit reached, stopping processing")
                break
                
            context = await self.memory.query(chunk.text)
            messages = build_summarization_prompt(chunk.text, context, self.config)
            
            response = await self._get_completion(messages)
            current_cost += response.cost
            summaries.append(response.content)
            
        # Combine chunk summaries into final summary
        if not summaries:
            raise ValueError("No text chunks were processed")
            
        final_summary = await self._merge_summaries(summaries)
        return final_summary
    
    async def _merge_summaries(self, summaries: List[str]) -> StructuredSummary:
        # Create a prompt to merge multiple summaries
        merge_prompt = f"""Merge these {len(summaries)} section summaries into a single coherent summary. 
        Maintain the same structured format but combine overlapping information and resolve any contradictions:
        
        {'\n\n'.join(summaries)}"""
        
        messages = [
            {"role": "system", "content": self.config.structured_summary_prompt},
            {"role": "user", "content": merge_prompt}
        ]
        
        response = await self._get_completion(messages)
        # Parse and validate the merged summary
        return self._parse_summary(response.content)
    
    def _estimate_cost(self, num_tokens: int) -> float:
        input_rate, output_rate = self.config.model_price[self.config.model]
        # Assume output is roughly same size as input for estimation
        return (num_tokens / 1000) * (input_rate + output_rate)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup resources
        if hasattr(self, 'memory'):
            del self.memory.index
        if hasattr(self, 'text_splitter'):
            del self.text_splitter.tokenizer
    
    async def process_large_text(self, text: str) -> AsyncIterator[TextChunk]:
        chunks = self.text_splitter.split_text(text)
        for chunk in chunks:
            yield chunk

    async def _parse_summary(self, summary_text: str) -> StructuredSummary:
        try:
            return StructuredSummary.from_text(summary_text)
        except ValidationError as e:
            logger.error(f"Failed to parse summary: {e}")
            raise
