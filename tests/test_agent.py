from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ecology_summarizer import AgentConfig, StructuredSummary, SummarizationAgent

FAKE_SUMMARY = StructuredSummary(
    title="Effects of Urbanization on Pollinator Networks",
    study_context="Urban expansion disrupts plant-pollinator interactions.",
    methods="Field surveys across an urban-rural gradient with network analysis.",
    key_findings="Urban sites showed 40% lower network connectivity.",
    ecological_implications="Urban planning should preserve pollinator corridors.",
    key_sentence="Urbanization significantly reduces pollinator network connectivity.",
)

FAKE_TEXT = "Fake paper content about pollinators and urban ecology."


def _make_agent(**config_overrides):
    with patch("ecology_summarizer.agent.instructor") as mock_inst:
        mock_inst.from_litellm.return_value.create = AsyncMock(return_value=FAKE_SUMMARY)
        agent = SummarizationAgent(AgentConfig(**config_overrides))
    return agent


@patch("ecology_summarizer.agent.extract_text", return_value=FAKE_TEXT)
@patch("ecology_summarizer.agent.instructor")
async def test_summarize_pdf_with_references(mock_instructor, mock_extract):
    mock_create = AsyncMock(return_value=FAKE_SUMMARY)
    mock_instructor.from_litellm.return_value.create = mock_create

    agent = SummarizationAgent()

    with patch("ecology_summarizer.agent.VectorMemory") as mock_memory_cls:
        mock_memory = AsyncMock()
        mock_memory.query.return_value = ["relevant context"]
        mock_memory_cls.return_value = mock_memory

        summary = await agent.summarize_pdf("fake.pdf", references=["ref1"])

    assert summary.title == "Effects of Urbanization on Pollinator Networks"
    mock_extract.assert_called_once_with("fake.pdf")
    mock_memory.add.assert_called_once()
    mock_memory.query.assert_called_once()
    mock_create.assert_called_once()


@patch("ecology_summarizer.agent.extract_text", return_value=FAKE_TEXT)
@patch("ecology_summarizer.agent.instructor")
async def test_summarize_pdf_no_references(mock_instructor, mock_extract):
    mock_create = AsyncMock(return_value=FAKE_SUMMARY)
    mock_instructor.from_litellm.return_value.create = mock_create

    agent = SummarizationAgent()

    with patch("ecology_summarizer.agent.VectorMemory") as mock_memory_cls:
        summary = await agent.summarize_pdf("fake.pdf")

    assert summary.title == "Effects of Urbanization on Pollinator Networks"
    mock_memory_cls.assert_not_called()
    mock_create.assert_called_once()


@patch("ecology_summarizer.agent.extract_text", return_value="")
@patch("ecology_summarizer.agent.instructor")
async def test_empty_pdf_raises(mock_instructor, mock_extract):
    agent = SummarizationAgent()
    with pytest.raises(ValueError, match="no extractable text"):
        await agent.summarize_pdf("empty.pdf")


@patch("ecology_summarizer.agent.extract_text")
@patch("ecology_summarizer.agent.instructor")
async def test_long_text_truncated(mock_instructor, mock_extract):
    mock_create = AsyncMock(return_value=FAKE_SUMMARY)
    mock_instructor.from_litellm.return_value.create = mock_create
    mock_extract.return_value = "x" * 5000

    agent = SummarizationAgent(AgentConfig(max_input_chars=100))
    await agent.summarize_pdf("big.pdf")

    call_args = mock_create.call_args
    user_msg = call_args.kwargs["messages"][1]["content"]
    paper_text = user_msg.split("Paper:\n\n")[1]
    assert len(paper_text) == 100


@patch("ecology_summarizer.pdf.fitz")
def test_pdf_document_closed(mock_fitz):
    mock_doc = MagicMock()
    mock_doc.__enter__ = MagicMock(return_value=mock_doc)
    mock_doc.__exit__ = MagicMock(return_value=False)
    mock_doc.__iter__ = MagicMock(return_value=iter([]))
    mock_fitz.open.return_value = mock_doc

    from ecology_summarizer.pdf import extract_text

    extract_text("test.pdf")
    mock_doc.__exit__.assert_called_once()


def test_config_validation():
    with pytest.raises(ValueError):
        AgentConfig(temperature=2.0)
    with pytest.raises(ValueError):
        AgentConfig(chunk_size=-1)
    with pytest.raises(ValueError):
        AgentConfig(max_input_chars=0)

    config = AgentConfig(model="anthropic/claude-sonnet-4-6", temperature=0.3)
    assert config.model == "anthropic/claude-sonnet-4-6"
    assert config.temperature == 0.3


def test_structured_summary_fields():
    data = FAKE_SUMMARY.model_dump()
    assert set(data.keys()) == {
        "title",
        "study_context",
        "methods",
        "key_findings",
        "ecological_implications",
        "key_sentence",
    }
