from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lit_review import (
    AgentConfig,
    EnvironmentalPredictors,
    EvaluationMethod,
    ExtractedField,
    ExtractionEval,
    FieldVerification,
    ModelingMethod,
    OccurrenceData,
    SDMExtractionAgent,
    SDMRequirements,
    SDMResults,
    StudyMetadata,
)
from lit_review.agent import _build_eval_content, _retrieval_query
from lit_review.prompts import EVAL_EXTRACTION_PREFIX, EVAL_PAPER_PREFIX

FAKE_REQUIREMENTS = SDMRequirements(
    study=StudyMetadata(
        title="Predicting Habitat Suitability for Quercus robur Under Climate Change",
        species=ExtractedField(
            value="Quercus robur",
            evidence="The study modeled Quercus robur distribution.",
            section="Methods",
            page=2,
        ),
        geographic_extent=ExtractedField(value="Western Europe (35-60N, 10W-25E)"),
    ),
    occurrence=OccurrenceData(
        occurrence_type=ExtractedField(value="presence-absence"),
        sample_size=ExtractedField(value="1,247 presence / 3,000 pseudo-absence"),
        occurrence_source=ExtractedField(value="GBIF, national forest inventories"),
    ),
    predictors=EnvironmentalPredictors(
        variables=ExtractedField(value="BIO1, BIO12, BIO15, elevation, soil pH"),
        source=ExtractedField(value="WorldClim v2.1, SoilGrids 250m"),
        spatial_resolution=ExtractedField(value="30 arc-seconds (~1 km)"),
    ),
    model=ModelingMethod(
        algorithm=ExtractedField(value="MaxEnt"),
        software=ExtractedField(value="R dismo package v1.3-9"),
        hyperparameters=ExtractedField(
            value="regularization multiplier = 1.5, feature classes = LQH, background points = 10000"
        ),
    ),
    evaluation=EvaluationMethod(
        metrics=ExtractedField(value="AUC = 0.92, TSS = 0.81"),
        cv_strategy=ExtractedField(value="Spatial block cross-validation, 5 folds"),
        threshold_method=ExtractedField(value="Maximum sensitivity + specificity"),
        performance_values=ExtractedField(value="Mean AUC = 0.92 +/- 0.03 across folds"),
    ),
    results=SDMResults(
        key_predictors=ExtractedField(
            value=(
                "BIO1 (mean annual temperature) and BIO12 (annual precipitation) "
                "contributed 72% of model gain"
            )
        ),
        predicted_distribution=ExtractedField(
            value=(
                "Suitable habitat projected to decline 35% under SSP5-8.5 by 2070, "
                "shifting northward into Scandinavia"
            )
        ),
    ),
)

FAKE_EVAL = ExtractionEval(
    field_verifications=[
        FieldVerification(
            field_path="study.species",
            extracted_value="Quercus robur",
            status="verified",
            evidence="The study modeled Quercus robur distribution",
        ),
        FieldVerification(
            field_path="model.algorithm",
            extracted_value="MaxEnt",
            status="verified",
            evidence="Models were fit using MaxEnt",
        ),
    ],
    num_verified=2,
    num_inaccurate=0,
    num_unverifiable=0,
    overall_assessment="Extraction accurately reflects the paper.",
)

FAKE_TEXT = "Fake paper content about oak species distribution modeling."


# ---------------------------------------------------------------------------
# Extraction tests
# ---------------------------------------------------------------------------


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

    assert (
        result.study.title
        == "Predicting Habitat Suitability for Quercus robur Under Climate Change"
    )
    assert result.study.species.value == "Quercus robur"
    assert result.study.species.evidence is not None
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

    assert (
        result.study.title
        == "Predicting Habitat Suitability for Quercus robur Under Climate Change"
    )
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
        "study",
        "occurrence",
        "predictors",
        "model",
        "evaluation",
        "results",
    }
    assert set(data["study"].keys()) == {"title", "species", "geographic_extent"}
    assert set(data["model"].keys()) == {"algorithm", "software", "hyperparameters"}


def test_sdm_requirements_optional_fields():
    minimal = SDMRequirements(study=StudyMetadata(title="A paper title"))
    assert minimal.study.title == "A paper title"
    assert minimal.study.species.value is None
    assert minimal.model.algorithm.value is None
    assert minimal.evaluation.metrics.value is None
    data = minimal.model_dump()
    assert data["study"]["species"]["value"] is None
    assert data["model"]["algorithm"]["value"] is None


def test_extracted_field_carries_evidence_location():
    field = ExtractedField(
        value="MaxEnt",
        evidence="We fit MaxEnt models using ENMeval.",
        section="Species distribution modeling",
        page=5,
    )
    assert field.value == "MaxEnt"
    assert field.evidence is not None
    assert field.section == "Species distribution modeling"
    assert field.page == 5


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------


@patch("lit_review.agent.extract_text", return_value=FAKE_TEXT)
@patch("lit_review.agent.instructor")
async def test_evaluate_calls_llm_with_eval_prompt(mock_instructor, mock_extract):
    mock_create = AsyncMock(return_value=FAKE_EVAL)
    mock_instructor.from_litellm.return_value.create = mock_create

    agent = SDMExtractionAgent()
    result = await agent.evaluate(FAKE_REQUIREMENTS, "fake.pdf")

    assert result.num_verified == 2
    assert result.num_inaccurate == 0
    assert len(result.field_verifications) == 2
    assert result.field_verifications[0].status == "verified"

    call_args = mock_create.call_args
    assert call_args.kwargs["response_model"] is ExtractionEval
    user_msg = call_args.kwargs["messages"][1]["content"]
    assert EVAL_EXTRACTION_PREFIX in user_msg
    assert FAKE_TEXT in user_msg


@patch("lit_review.agent.extract_text", return_value="")
@patch("lit_review.agent.instructor")
async def test_evaluate_empty_pdf_raises(mock_instructor, mock_extract):
    agent = SDMExtractionAgent()
    with pytest.raises(ValueError, match="no extractable text"):
        await agent.evaluate(FAKE_REQUIREMENTS, "empty.pdf")


def test_eval_content_respects_max_chars():
    req_json = '{"title": "Test"}'
    paper = "x" * 5000
    content = _build_eval_content(req_json, paper, max_chars=200)
    assert len(content) == 200
    assert content.startswith(EVAL_EXTRACTION_PREFIX)


def test_eval_content_includes_both_sections():
    req_json = '{"title": "Test", "algorithm": "MaxEnt"}'
    paper = "This paper describes a MaxEnt species distribution model."
    content = _build_eval_content(req_json, paper, max_chars=10_000)
    assert EVAL_EXTRACTION_PREFIX in content
    assert req_json in content
    assert paper in content


def test_field_verification_model():
    fv = FieldVerification(
        field_path="model.algorithm",
        extracted_value="MaxEnt",
        status="verified",
        evidence="The authors used MaxEnt v3.4.4",
    )
    assert fv.status == "verified"
    assert fv.evidence is not None

    fv_no_evidence = FieldVerification(
        field_path="model.software",
        extracted_value="R dismo",
        status="unverifiable",
    )
    assert fv_no_evidence.evidence is None


def test_extraction_eval_model():
    eval_result = ExtractionEval(
        field_verifications=[
            FieldVerification(
                field_path="study.species",
                extracted_value="Quercus robur",
                status="verified",
                evidence="Study species was Q. robur",
            ),
            FieldVerification(
                field_path="occurrence.sample_size",
                extracted_value="500 points",
                status="inaccurate",
                evidence="Paper states 350 presence records",
            ),
        ],
        num_verified=1,
        num_inaccurate=1,
        num_unverifiable=0,
        overall_assessment="One field inaccurate: sample size mismatch.",
    )
    assert len(eval_result.field_verifications) == 2
    assert eval_result.num_verified == 1
    assert eval_result.num_inaccurate == 1
