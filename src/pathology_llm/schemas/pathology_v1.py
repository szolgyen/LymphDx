from pydantic import BaseModel, Field
from typing import Optional, List


class Biomarker(BaseModel):
    name: str
    value: Optional[str] = None


class PathologyExtraction(BaseModel):
    schema_version: str = Field(default="v1", frozen=True)

    diagnosis_primary: Optional[str] = None
    diagnosis_secondary: List[str] = Field(default_factory=list)

    description: Optional[str] = None
    interpretation_status: Optional[str] = None

    specimen: Optional[str] = None
    is_lymph_node: Optional[bool] = None
    anatomic_location: Optional[str] = None

    container: Optional[str] = None

    biomarkers: List[Biomarker] = Field(default_factory=list)

    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
