import pytest
from unittest.mock import AsyncMock, patch
from ecology_summarizer import SummarizationAgent, AgentConfig, StructuredSummary


FAKE_SUMMARY = StructuredSummary(
    title="Effects of Urbanization on Pollinator Networks",
    study_context="Urban expansion disrupts plant-pollinator interactions.",
    methods="Field surveys across an urban-rural gradient with network analysis.",
    key_findings="Urban sites showed 40% lower network connectivity.",
    ecological_implications="Urban planning should preserve pollinator corridors.",
    key_sentence="Urbanization significantly reduces pollinator network connectivity, suggesting corridor preservation is critical.",
)


@patch("ecology_summarizer.agent.extract_text", return_value="Fake paper content about pollinators.")
@patch("ecology_summarizer.agent.instructor")
async def test_summarize_pdf(mock_instructor, mock_extract):
    mock_create = AsyncMock(return_value=FAKE_SUMMARY)
    mock_instructor.from_litellm.return_value.create = mock_create

    agent = SummarizationAgent()
    summary = await agent.summarize_pdf("fake.pdf")

    assert summary.title == "Effects of Urbanization on Pollinator Networks"
    assert "pollinator" in summary.key_sentence.lower()
    mock_extract.assert_called_once_with("fake.pdf")
    mock_create.assert_called_once()


def test_config_validation():
    with pytest.raises(ValueError):
        AgentConfig(temperature=2.0)

    with pytest.raises(ValueError):
        AgentConfig(chunk_size=-1)

    config = AgentConfig(model="anthropic/claude-sonnet-4-6", temperature=0.3)
    assert config.model == "anthropic/claude-sonnet-4-6"
    assert config.temperature == 0.3


def test_structured_summary_fields():
    data = FAKE_SUMMARY.model_dump()
    assert set(data.keys()) == {
        "title", "study_context", "methods",
        "key_findings", "ecological_implications", "key_sentence",
    }
