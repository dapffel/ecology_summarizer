from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    model: str = "gpt-4"
    embedding_model: str = "text-embedding-ada-002"
    temperature: float = Field(default=0.2, ge=0, le=1)
    max_reference_docs: int = Field(default=10, gt=0)
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    max_input_chars: int = Field(default=100_000, gt=0)


class SDMRequirements(BaseModel):
    title: str = Field(description="Title of the paper")
    species: str | None = Field(
        default=None,
        description=(
            "Focal species or taxonomic group modeled "
            "(e.g. 'Quercus robur', 'European bats', 'invasive plants'). "
            "Use the scientific name when given."
        ),
    )
    geographic_extent: str | None = Field(
        default=None,
        description=(
            "Geographic scope of the study area "
            "(e.g. 'Iberian Peninsula', 'global', 'eastern United States'). "
            "Include coordinates or bounding box if reported."
        ),
    )
    occurrence_type: str | None = Field(
        default=None,
        description=(
            "Type of species occurrence data used: "
            "'presence-only', 'presence-absence', or 'abundance'."
        ),
    )
    sample_size: str | None = Field(
        default=None,
        description=(
            "Number of occurrence records or sites used for modeling "
            "(e.g. '342 presence points', '150 presence / 200 absence'). "
            "Include both training and test sizes if reported separately."
        ),
    )
    occurrence_source: str | None = Field(
        default=None,
        description=(
            "Source of occurrence data "
            "(e.g. 'GBIF', 'field surveys 2010-2020', 'museum specimens', 'eBird'). "
            "List all sources if multiple."
        ),
    )
    environmental_variables: str | None = Field(
        default=None,
        description=(
            "List of environmental predictor variables used in the model "
            "(e.g. 'bioclimatic variables BIO1-BIO19, elevation, land cover, NDVI'). "
            "Be specific about which variables were retained in the final model if reported."
        ),
    )
    environmental_source: str | None = Field(
        default=None,
        description=(
            "Source datasets for environmental predictors "
            "(e.g. 'WorldClim v2.1', 'CHELSA', 'MODIS'). "
            "Include version numbers when provided."
        ),
    )
    spatial_resolution: str | None = Field(
        default=None,
        description=(
            "Spatial resolution or grid cell size of the environmental data "
            "and/or model predictions (e.g. '30 arc-seconds (~1 km)', '250 m'). "
            "Note the coordinate reference system if reported."
        ),
    )
    algorithm: str | None = Field(
        default=None,
        description=(
            "SDM algorithm(s) used "
            "(e.g. 'MaxEnt', 'Random Forest', 'GLM', 'BRT', "
            "'ensemble of MaxEnt + BRT + GAM'). "
            "List all algorithms if a comparison or ensemble study."
        ),
    )
    software: str | None = Field(
        default=None,
        description=(
            "Software, package, or platform used to fit the SDM "
            "(e.g. 'R dismo package', 'MaxEnt 3.4.4', 'biomod2', 'ENMeval'). "
            "Include version numbers when provided."
        ),
    )
    hyperparameters: str | None = Field(
        default=None,
        description=(
            "Key model hyperparameters or settings reported "
            "(e.g. 'regularization multiplier = 2, feature classes = LQH', "
            "'ntree = 1000, mtry = 4', 'background points = 10000'). "
            "Include feature selection method if reported."
        ),
    )
    evaluation_metrics: str | None = Field(
        default=None,
        description=(
            "Performance metrics reported "
            "(e.g. 'AUC = 0.92, TSS = 0.78', 'Boyce index', 'RMSE'). "
            "Include numeric values when available."
        ),
    )
    cv_strategy: str | None = Field(
        default=None,
        description=(
            "Cross-validation or data-splitting strategy used for evaluation "
            "(e.g. '10-fold cross-validation', 'spatial block CV', '70/30 random split'). "
            "Note if spatial structure was accounted for."
        ),
    )
    threshold_method: str | None = Field(
        default=None,
        description=(
            "Method used to convert continuous suitability to binary predictions "
            "(e.g. 'maximum sensitivity + specificity', "
            "'10th percentile training presence'). "
            "State 'not reported' if the paper does not specify."
        ),
    )
    performance_values: str | None = Field(
        default=None,
        description=(
            "Summary of the best or final model performance values "
            "(e.g. 'mean AUC = 0.91 +/- 0.03 across folds'). "
            "Report as precisely as the paper states."
        ),
    )
    key_predictors: str | None = Field(
        default=None,
        description=(
            "Most important predictor variables identified by the model "
            "(e.g. 'mean annual temperature (BIO1) and precipitation seasonality (BIO15) "
            "contributed 65% of model gain'). "
            "Include variable importance values if reported."
        ),
    )
    predicted_distribution: str | None = Field(
        default=None,
        description=(
            "Brief description of the predicted distribution or suitability map, "
            "including projected range shifts, area of suitable habitat, "
            "or future climate scenarios used "
            "(e.g. 'suitable habitat projected to shift 300 km northward "
            "under SSP3-7.0 by 2070')."
        ),
    )
