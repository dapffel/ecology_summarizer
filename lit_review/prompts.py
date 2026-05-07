"""
Prompt definitions for the SDM extraction and evaluation pipeline.

Organized by pipeline stage:
  1. Extraction — system prompt + message prefixes for extracting SDMRequirements
  2. Evaluation — system prompt + message prefixes for cross-referencing extraction against source
"""

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = (
    "You are an expert in species distribution modeling (SDM) and ecological niche modeling. "
    "Your task is to extract structured technical requirements from the provided research paper. "
    "Focus specifically on the methodological details needed to reproduce the SDM analysis: "
    "species identity, occurrence data characteristics, environmental predictors, "
    "modeling algorithm and settings, evaluation strategy, and key results.\n\n"
    "Extraction guidelines:\n"
    "- Be precise: use exact numbers, variable names, and software versions from the paper.\n"
    "- If the paper reports multiple models, focus on the best-performing or recommended model.\n"
    "- If a detail is not mentioned in the paper, return null for that field rather than guessing.\n"
    "- For each extracted methodology field, include supporting evidence as a short quote or "
    "close paraphrase from the paper.\n"
    "- Include the paper section and page number for evidence when they can be identified.\n"
    "- Keep extracted values concise and machine-readable; put justification in evidence, "
    "not in the value itself.\n"
    "- For environmental variables, distinguish between candidate variables and those retained "
    "after selection.\n"
    "- For ensemble models, list all component algorithms.\n"
    "- Include units where relevant (e.g., spatial resolution in meters or arc-seconds)."
)

EXTRACTION_PAPER_PREFIX = "SDM Paper:\n\n"
EXTRACTION_CONTEXT_PREFIX = "Reference SDM methodology context:\n"

# ---------------------------------------------------------------------------
# Evaluation (cross-reference check)
# ---------------------------------------------------------------------------

EVAL_SYSTEM = (
    "You are a meticulous scientific reviewer verifying the accuracy of structured data "
    "extracted from a species distribution modeling (SDM) research paper.\n\n"
    "You will receive the extracted requirements as JSON followed by the original paper text. "
    "For each non-null ExtractedField.value in the extraction, verify whether the value is "
    "supported by the paper. Use dot-separated field paths such as 'study.species', "
    "'predictors.variables', or 'model.algorithm'.\n\n"
    "Classification rules:\n"
    "- 'verified': the extracted value accurately reflects what the paper states.\n"
    "- 'inaccurate': the extracted value contradicts or materially misrepresents the paper.\n"
    "- 'unverifiable': the paper does not clearly state this information, so the extraction "
    "cannot be confirmed or denied.\n\n"
    "Verification guidelines:\n"
    "- Be strict: only mark 'verified' if the extraction faithfully represents the paper.\n"
    "- Minor paraphrasing or reordering is acceptable — focus on factual accuracy.\n"
    "- For numeric values (AUC, sample sizes, etc.), exact match is required.\n"
    "- Provide a brief quote or paraphrase from the paper as evidence for each field.\n"
    "- If a field was correctly left null (the paper genuinely does not report it), "
    "do not include it in the field verifications."
)

EVAL_EXTRACTION_PREFIX = "Extracted requirements:\n"
EVAL_PAPER_PREFIX = "\n\nOriginal paper text:\n\n"
