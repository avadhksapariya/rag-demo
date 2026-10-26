from pydantic import BaseModel, Field
from typing import List


class SentenceVerification(BaseModel):
    sentence: str = Field(description="The exact sentence from the generated answer.")
    is_supported: bool = Field(
        description="True if the facts in this sentence are directly corroborated by the retrieved context. False otherwise."
    )
    unsupported_reason: str = Field(
        default="",
        description="Brief explanation if unsupported or extrapolated. Empty if supported.",
    )


class GroundingReport(BaseModel):
    verifications: List[SentenceVerification] = Field(
        description="Verification result for each factual sentence."
    )
    overall_score: float = Field(
        description="Ratio of supported sentences over total factual sentences (between 0.0 and 1.0)."
    )
    needs_correction: bool = Field(
        description="True if any critical sentence is unsupported; False if all are verified."
    )
