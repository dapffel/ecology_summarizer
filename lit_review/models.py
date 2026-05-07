from __future__ import annotations

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    model: str = "gpt-4"
    embedding_model: str = "text-embedding-ada-002"
    temperature: float = Field(default=0.2, ge=0, le=1)
    max_reference_docs: int = Field(default=10, gt=0)
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    max_input_chars: int = Field(default=100_000, gt=0)


class ExtractedField(BaseModel):
    value: str | None = Field(
        default=None,
        description="Extracted value, or null if the paper does not report this detail.",
    )
    evidence: str | None = Field(
        default=None,
        description="Brief quote or close paraphrase supporting the extracted value.",
    )
    section: str | None = Field(
        default=None,
        description="Paper section where the evidence appears, if identifiable.",
    )
    page: int | None = Field(
        default=None,
        ge=1,
        description="PDF page number where the evidence appears, if identifiable.",
    )


class StudyMetadata(BaseModel):
    title: str = Field(description="Title of the paper")
    species: ExtractedField = Field(
        default_factory=ExtractedField,
        description=(
            "Focal species or taxonomic group modeled "
            "(e.g. 'Quercus robur', 'European bats', 'invasive plants'). "
            "Use the scientific name when given."
        ),
    )
    geographic_extent: ExtractedField = Field(
        default_factory=ExtractedField,
        description=(
            "Geographic scope of the study area "
            "(e.g. 'Iberian Peninsula', 'global', 'eastern United States'). "
            "Include coordinates or bounding box if reported."
        ),
    )


class OccurrenceData(BaseModel):
    occurrence_type: ExtractedField = Field(
        default_factory=ExtractedField,
        description=(
            "Type of species occurrence data used: "
            "'presence-only', 'presence-absence', or 'abundance'."
        ),
    )
    sample_size: ExtractedField = Field(
        default_factory=ExtractedField,
        description=(
            "Number of occurrence records or sites used for modeling "
            "(e.g. '342 presence points', '150 presence / 200 absence'). "
            "Include both training and test sizes if reported separately."
        ),
    )
    occurrence_source: ExtractedField = Field(
        default_factory=ExtractedField,
        description=(
            "Source of occurrence data "
            "(e.g. 'GBIF', 'field surveys 2010-2020', 'museum specimens', 'eBird'). "
            "List all sources if multiple."
        ),
    )


class EnvironmentalPredictors(BaseModel):
    variables: ExtractedField = Field(
        default_factory=ExtractedField,
        description=(
            "List of environmental predictor variables used in the model "
            "(e.g. 'bioclimatic variables BIO1-BIO19, elevation, land cover, NDVI'). "
            "Be specific about which variables were retained in the final model if reported."
        ),
    )
    source: ExtractedField = Field(
        default_factory=ExtractedField,
        description=(
            "Source datasets for environmental predictors "
            "(e.g. 'WorldClim v2.1', 'CHELSA', 'MODIS'). "
            "Include version numbers when provided."
        ),
    )
    spatial_resolution: ExtractedField = Field(
        default_factory=ExtractedField,
        description=(
            "Spatial resolution or grid cell size of the environmental data "
            "and/or model predictions (e.g. '30 arc-seconds (~1 km)', '250 m'). "
            "Note the coordinate reference system if reported."
        ),
    )


class ModelingMethod(BaseModel):
    algorithm: ExtractedField = Field(
        default_factory=ExtractedField,
        description=(
            "SDM algorithm(s) used "
            "(e.g. 'MaxEnt', 'Random Forest', 'GLM', 'BRT', "
            "'ensemble of MaxEnt + BRT + GAM'). "
            "List all algorithms if a comparison or ensemble study."
        ),
    )
    software: ExtractedField = Field(
        default_factory=ExtractedField,
        description=(
            "Software, package, or platform used to fit the SDM "
            "(e.g. 'R dismo package', 'MaxEnt 3.4.4', 'biomod2', 'ENMeval'). "
            "Include version numbers when provided."
        ),
    )
    hyperparameters: ExtractedField = Field(
        default_factory=ExtractedField,
        description=(
            "Key model hyperparameters or settings reported "
            "(e.g. 'regularization multiplier = 2, feature classes = LQH', "
            "'ntree = 1000, mtry = 4', 'background points = 10000'). "
            "Include feature selection method if reported."
        ),
    )


class EvaluationMethod(BaseModel):
    metrics: ExtractedField = Field(
        default_factory=ExtractedField,
        description=(
            "Performance metrics reported "
            "(e.g. 'AUC = 0.92, TSS = 0.78', 'Boyce index', 'RMSE'). "
            "Include numeric values when available."
        ),
    )
    cv_strategy: ExtractedField = Field(
        default_factory=ExtractedField,
        description=(
            "Cross-validation or data-splitting strategy used for evaluation "
            "(e.g. '10-fold cross-validation', 'spatial block CV', '70/30 random split'). "
            "Note if spatial structure was accounted for."
        ),
    )
    threshold_method: ExtractedField = Field(
        default_factory=ExtractedField,
        description=(
            "Method used to convert continuous suitability to binary predictions "
            "(e.g. 'maximum sensitivity + specificity', "
            "'10th percentile training presence'). "
            "Leave value null if the paper does not specify."
        ),
    )
    performance_values: ExtractedField = Field(
        default_factory=ExtractedField,
        description=(
            "Summary of the best or final model performance values "
            "(e.g. 'mean AUC = 0.91 +/- 0.03 across folds'). "
            "Report as precisely as the paper states."
        ),
    )


class SDMResults(BaseModel):
    key_predictors: ExtractedField = Field(
        default_factory=ExtractedField,
        description=(
            "Most important predictor variables identified by the model "
            "(e.g. 'mean annual temperature (BIO1) and precipitation seasonality (BIO15) "
            "contributed 65% of model gain'). "
            "Include variable importance values if reported."
        ),
    )
    predicted_distribution: ExtractedField = Field(
        default_factory=ExtractedField,
        description=(
            "Brief description of the predicted distribution or suitability map, "
            "including projected range shifts, area of suitable habitat, "
            "or future climate scenarios used "
            "(e.g. 'suitable habitat projected to shift 300 km northward "
            "under SSP3-7.0 by 2070')."
        ),
    )


class SDMRequirements(BaseModel):
    study: StudyMetadata = Field(description="Study metadata and focal modeling target")
    occurrence: OccurrenceData = Field(
        default_factory=OccurrenceData,
        description="Occurrence data and sampling details needed to recreate the SDM",
    )
    predictors: EnvironmentalPredictors = Field(
        default_factory=EnvironmentalPredictors,
        description="Environmental predictor data used by the SDM",
    )
    model: ModelingMethod = Field(
        default_factory=ModelingMethod,
        description="Modeling algorithm, software, and settings",
    )
    evaluation: EvaluationMethod = Field(
        default_factory=EvaluationMethod,
        description="Validation, thresholding, and performance reporting",
    )
    results: SDMResults = Field(
        default_factory=SDMResults,
        description="Reported predictor importance and distribution outputs",
    )


class FieldVerification(BaseModel):
    field_path: str = Field(
        description=(
            "Dot-separated path to the SDMRequirements field being verified "
            "(e.g. 'model.algorithm')"
        )
    )
    extracted_value: str = Field(description="The value that was extracted for this field")
    status: str = Field(
        description=(
            "'verified' if the extraction accurately reflects the paper, "
            "'inaccurate' if it contradicts or misrepresents the paper, "
            "'unverifiable' if the paper does not clearly state this information"
        ),
    )
    evidence: str | None = Field(
        default=None,
        description=(
            "Brief quote or paraphrase from the paper "
            "supporting or contradicting the extracted value"
        ),
    )


class ExtractionEval(BaseModel):
    field_verifications: list[FieldVerification] = Field(
        description="Verification result for each non-null extracted field"
    )
    num_verified: int = Field(description="Count of fields classified as verified")
    num_inaccurate: int = Field(description="Count of fields classified as inaccurate")
    num_unverifiable: int = Field(description="Count of fields classified as unverifiable")
    overall_assessment: str = Field(
        description="Brief overall assessment of extraction quality and any key issues found"
    )
