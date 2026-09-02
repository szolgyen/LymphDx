from pydantic import BaseModel, Field
from typing import Literal, Optional, List


class Biomarker(BaseModel):
    name: str
    value: Optional[str] = None


class ContainerInfo(BaseModel):
    label: Optional[str] = None
    specimen: Optional[str] = None
    is_lymph_node: Optional[bool] = None
    anatomic_location: Optional[str] = None
    diagnosis: Optional[str] = None


class ContainerInfo_v2(BaseModel):
    label: Optional[str] = None
    is_lymph_node: Optional[bool] = None
    diagnosis: Optional[str] = Field(
        default=None,
        json_schema_extra={"x-reportllm-diagnosis-constrained": True},
    )


class PathologyExtractionV2(BaseModel):
    schema_version: Literal["v2"] = Field(default="v2", frozen=True)

    case_id: Optional[int] = Field(
        default=None,
        json_schema_extra={"x-reportllm-decoder-exclude": True},
    )

    primary_diagnosis: Optional[str] = Field(
        default=None,
        json_schema_extra={"x-reportllm-diagnosis-constrained": True},
    )

    has_differential_diagnosis: Optional[bool] = None
    differential_diagnoses: List[str] = Field(
        default_factory=list,
        max_length=10,
        json_schema_extra={"x-reportllm-diagnosis-constrained": True},
    )

    specimen: Optional[str] = None
    is_lymph_node: Optional[bool] = None
    is_definitive: Optional[bool] = None

    has_prior_malignancy: Optional[bool] = None
    has_concurrent_malignancy: Optional[bool] = None
    anatomic_location: Optional[str] = None

    container: Optional[str] = None

    biomarkers: List[Biomarker] = Field(default_factory=list)

    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class PathologyExtractionV3(BaseModel):
    schema_version: Literal["v3"] = Field(default="v3", frozen=True)

    case_id: Optional[int] = Field(
        default=None,
        json_schema_extra={"x-reportllm-decoder-exclude": True},
    )

    primary_diagnosis: Optional[str] = Field(
        default=None,
        json_schema_extra={"x-reportllm-diagnosis-constrained": True},
    )
    is_definitive: Optional[bool] = None
    containers: Optional[List[ContainerInfo]] = None
    has_differential_diagnosis: Optional[bool] = None
    has_prior_malignancy: Optional[bool] = None
    has_concurrent_malignancy: Optional[bool] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class PathologyExtractionV4(BaseModel):
    schema_version: Literal["v4"] = Field(default="v4", frozen=True)

    case_id: Optional[int] = Field(
        default=None,
        json_schema_extra={"x-reportllm-decoder-exclude": True},
    )

    primary_diagnosis: Optional[str] = Field(
        default=None,
        json_schema_extra={"x-reportllm-diagnosis-constrained": False},
    )


class PathologyExtractionV5(BaseModel):
    schema_version: Literal["v5"] = Field(default="v5", frozen=True)

    case_id: Optional[int] = Field(
        default=None,
        json_schema_extra={"x-reportllm-decoder-exclude": True},
    )

    primary_diagnosis: Optional[str] = Field(
        default=None,
        json_schema_extra={"x-reportllm-diagnosis-constrained": True},
    )
    is_definitive: Optional[bool] = None
    containers: Optional[List[ContainerInfo]] = None
    has_differential_diagnosis: Optional[bool] = None
    has_prior_malignancy: Optional[bool] = None
    has_concurrent_malignancy: Optional[bool] = None


class PathologyExtractionV6(BaseModel):
    schema_version: Literal["v6"] = Field(default="v6", frozen=True)

    case_id: Optional[int] = Field(
        default=None,
        json_schema_extra={"x-reportllm-decoder-exclude": True},
    )

    primary_diagnosis: Optional[str] = Field(
        default=None,
        json_schema_extra={"x-reportllm-diagnosis-constrained": True},
    )
    diagnosis_group_1: str = None
    diagnosis_group_2: str = None
    diagnosis_group_3: str = None


class PathologyExtractionV7(BaseModel):
    schema_version: Literal["v7"] = Field(default="v7", frozen=True)

    case_id: Optional[int] = Field(
        default=None,
        json_schema_extra={"x-reportllm-decoder-exclude": True},
    )

    primary_diagnosis: Optional[str] = Field(
        default=None,
        json_schema_extra={"x-reportllm-diagnosis-constrained": True},
    )
    is_definitive: Optional[bool] = None
    containers: Optional[List[ContainerInfo_v2]] = None
    has_differential_diagnosis: Optional[bool] = None
    has_prior_malignancy: Optional[bool] = None
    has_concurrent_malignancy: Optional[bool] = None


schema_v2 = PathologyExtractionV2
schema_v3 = PathologyExtractionV3
schema_v4 = PathologyExtractionV4
schema_v5 = PathologyExtractionV5
schema_v6 = PathologyExtractionV6
schema_v7 = PathologyExtractionV7
