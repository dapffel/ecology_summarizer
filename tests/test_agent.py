from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lit_review import AgentConfig, SDMExtractionAgent, SDMRequirements
from lit_review.agent import _retrieval_query

FAKE_REQUIREMENTS = SDMRequirements(
    title="Predicting Habitat Suitability for Quercus robur Under Climate Change",
    species="Quercus robur",
    geographic_extent="Western Europe (35-60N, 10W-25E)",
    occurrence_type="presence-absence",
    sample_size="1,247 presence / 3,000 pseudo-absence",
    occurrence_source="GBIF, national forest inventories",
    environmental_variables="BIO1, BIO12, BIO15, elevation, soil pH",
    environmental_source="WorldClim v2.1, SoilGrids 250m",
    spatial_resolution="30 arc-seconds (~1 km)",
    algorithm="MaxEnt",
    software="R dismo package v1.3-9",
    hyperparameters="regularization multiplier = 1.5, feature classes = LQH, background points = 10000",
    evaluation_metrics="AUC = 0.92, TSS = 0.81",
    cv_strategy="Spatial block cross-validation, 5 folds",
    threshold_method="Maximum sensitivity + specificity",
    performance_values="Mean AUC = 0.92 +/- 0.03 across folds",
    key_predictors="BIO1 (mean annual temperature) and BIO12 (annual precipitation) contributed 72% of model gain",
    predicted_distribution="Suitable habitat projected to decline 35% under SSP5-8.5 by 2070, shifting northward into Scandinavia",
)

FAKE_TEXT = "Fake paper content about oak species distribution modeling."


@patch("lit_review.agent.extract_text", return_value=FAKE_TEXT)
@patch("lit_review.agent.instructor")
async def test_extract_from_pdf_with_references(mock_instructor, mock_extract):
    mock_create = AsyncMock(return_value=FAKE_REQUIREMENTS)
    mock_instructor.from_litellm.return_value.create = mock_create

    agent = SDMExtractionAgent()

    with patch("lit_review.agent.VectorMemory") as mock_memory_cls:
        mock_memory = AsyncMock()
        mock_memory.query.return_value = ["relevant context"]
        mock_memory_cls.return_value = mock_memory

        result = await agent.extract_from_pdf("fake.pdf", references=["ref1"])

    assert result.title == "Predicting Habitat Suitability for Quercus robur Under Climate Change"
    assert result.species == "Quercus robur"
    mock_extract.assert_called_once_with("fake.pdf")
    mock_memory.add.assert_called_once()
    mock_memory.query.assert_called_once()
    mock_create.assert_called_once()


@patch("lit_review.agent.extract_text", return_value=FAKE_TEXT)
@patch("lit_review.agent.instructor")
async def test_extract_from_pdf_no_references(mock_instructor, mock_extract):
    mock_create = AsyncMock(return_value=FAKE_REQUIREMENTS)
    mock_instructor.from_litellm.return_value.create = mock_create

    agent = SDMExtractionAgent()

    with patch("lit_review.agent.VectorMemory") as mock_memory_cls:
        result = await agent.extract_from_pdf("fake.pdf")

    assert result.title == "Predicting Habitat Suitability for Quercus robur Under Climate Change"
    mock_memory_cls.assert_not_called()
    mock_create.assert_called_once()


@patch("lit_review.agent.extract_text", return_value="")
@patch("lit_review.agent.instructor")
async def test_empty_pdf_raises(mock_instructor, mock_extract):
    agent = SDMExtractionAgent()
    with pytest.raises(ValueError, match="no extractable text"):
        await agent.extract_from_pdf("empty.pdf")


@patch("lit_review.agent.extract_text")
@patch("lit_review.agent.instructor")
async def test_long_text_truncated(mock_instructor, mock_extract):
    mock_create = AsyncMock(return_value=FAKE_REQUIREMENTS)
    mock_instructor.from_litellm.return_value.create = mock_create
    mock_extract.return_value = "x" * 5000

    agent = SDMExtractionAgent(AgentConfig(max_input_chars=100))
    await agent.extract_from_pdf("big.pdf")

    call_args = mock_create.call_args
    user_msg = call_args.kwargs["messages"][1]["content"]
    assert len(user_msg) == 100


@patch("lit_review.agent.extract_text")
@patch("lit_review.agent.instructor")
async def test_prompt_with_context_respects_max_input_chars(mock_instructor, mock_extract):
    mock_create = AsyncMock(return_value=FAKE_REQUIREMENTS)
    mock_instructor.from_litellm.return_value.create = mock_create
    mock_extract.return_value = "x" * 5000

    agent = SDMExtractionAgent(AgentConfig(max_input_chars=200))

    with patch("lit_review.agent.VectorMemory") as mock_memory_cls:
        mock_memory = AsyncMock()
        mock_memory.query.return_value = ["context " * 20]
        mock_memory_cls.return_value = mock_memory

        await agent.extract_from_pdf("big.pdf", references=["ref1"])

    user_msg = mock_create.call_args.kwargs["messages"][1]["content"]
    assert len(user_msg) == 200


def test_retrieval_query_uses_first_substantial_paragraph():
    text = "Title\n\nShort.\n\n" + ("This paragraph has enough ecological detail. " * 4)
    query = _retrieval_query(text)
    assert query.startswith("This paragraph")


@patch("lit_review.pdf.fitz")
def test_pdf_document_closed(mock_fitz):
    mock_doc = MagicMock()
    mock_doc.__enter__ = MagicMock(return_value=mock_doc)
    mock_doc.__exit__ = MagicMock(return_value=False)
    mock_doc.__iter__ = MagicMock(return_value=iter([]))
    mock_fitz.open.return_value = mock_doc

    from lit_review.pdf import extract_text

    extract_text("test.pdf")
    mock_doc.__exit__.assert_called_once()


def test_config_validation():
    with pytest.raises(ValueError):
        AgentConfig(temperature=2.0)
    with pytest.raises(ValueError):
        AgentConfig(chunk_size=-1)
    with pytest.raises(ValueError):
        AgentConfig(max_input_chars=0)

    config = AgentConfig(model="anthropic/claude-sonnet-4-6", temperature=0.1)
    assert config.model == "anthropic/claude-sonnet-4-6"
    assert config.temperature == 0.1


def test_sdm_requirements_fields():
    data = FAKE_REQUIREMENTS.model_dump()
    assert set(data.keys()) == {
        "title",
        "species",
        "geographic_extent",
        "occurrence_type",
        "sample_size",
        "occurrence_source",
        "environmental_variables",
        "environmental_source",
        "spatial_resolution",
        "algorithm",
        "software",
        "hyperparameters",
        "evaluation_metrics",
        "cv_strategy",
        "threshold_method",
        "performance_values",
        "key_predictors",
        "predicted_distribution",
    }


def test_sdm_requirements_optional_fields():
    minimal = SDMRequirements(title="A paper title")
    assert minimal.title == "A paper title"
    assert minimal.species is None
    assert minimal.algorithm is None
    assert minimal.evaluation_metrics is None
    data = minimal.model_dump()
    none_fields = [k for k, v in data.items() if v is None]
    assert len(none_fields) == 17
