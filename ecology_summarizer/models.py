from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    model: str = "gpt-4"
    embedding_model: str = "text-embedding-ada-002"
    temperature: float = Field(default=0.5, ge=0, le=1)
    max_reference_docs: int = Field(default=10, gt=0)
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)


class StructuredSummary(BaseModel):
    title: str = Field(description="Title of the paper")
    study_context: str = Field(description="Ecological context and motivation for the study")
    methods: str = Field(description="Research methodology and approach")
    key_findings: str = Field(description="Main results and discoveries")
    ecological_implications: str = Field(description="Broader implications for ecology")
    key_sentence: str = Field(description="One-sentence summary suitable for a research proposal")
