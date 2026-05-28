from pydantic import BaseModel, Field
from typing import Literal, Optional, List


class Biomarker(BaseModel):
    name: str
    value: Optional[str] = None


class PathologyExtraction(BaseModel):
    schema_version: Literal["v2"] = Field(default="v2", frozen=True)

    case_id: Optional[int] = None

    primary_diagnosis: Optional[str] = None

    has_differential_diagnosis: Optional[bool] = None
    differential_diagnoses: List[str] = Field(default_factory=list)

    specimen: Optional[str] = None
    is_lymph_node: Optional[bool] = None
    is_definitive: Optional[bool] = None

    has_prior_malignancy: Optional[bool] = None
    has_concurrent_malignancy: Optional[bool] = None
    anatomic_location: Optional[str] = None

    container: Optional[str] = None

    biomarkers: List[Biomarker] = Field(default_factory=list)

    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
