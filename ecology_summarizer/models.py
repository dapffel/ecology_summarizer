from pydantic import BaseModel, Field


class StructuredSummary(BaseModel):
    title: str = Field(description="Title of the paper")
    study_context: str = Field(description="Ecological context and motivation for the study")
    methods: str = Field(description="Research methodology and approach")
    key_findings: str = Field(description="Main results and discoveries")
    ecological_implications: str = Field(description="Broader implications for ecology")
    key_sentence: str = Field(description="One-sentence summary suitable for a research proposal")
