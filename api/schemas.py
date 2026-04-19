"""
api/schemas.py

Pydantic schemas for the ML Pipeline Monitoring System.
Dataset: UCI Adult Income (predict whether income > $50K)

These schemas are the single source of truth for:
- Input validation on the /predict endpoint
- Response shapes returned to clients
- Logged prediction records written to PostgreSQL
- Drift detection baseline (Evidently compares live traffic against these field definitions)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations — strict categorical constraints
# Any value outside these sets is rejected at the API boundary, before it
# ever reaches the model. This is intentional: silent garbage-in → garbage-out
# is the #1 cause of silent model degradation in production.
# ---------------------------------------------------------------------------

class WorkclassEnum(str, Enum):
    PRIVATE         = "Private"
    SELF_EMP_NOT_INC = "Self-emp-not-inc"
    SELF_EMP_INC    = "Self-emp-inc"
    FEDERAL_GOV     = "Federal-gov"
    LOCAL_GOV       = "Local-gov"
    STATE_GOV       = "State-gov"
    WITHOUT_PAY     = "Without-pay"
    NEVER_WORKED    = "Never-worked"


class EducationEnum(str, Enum):
    BACHELORS       = "Bachelors"
    SOME_COLLEGE    = "Some-college"
    ELEVENTH        = "11th"
    HS_GRAD         = "HS-grad"
    PROF_SCHOOL     = "Prof-school"
    ASSOC_ACDM      = "Assoc-acdm"
    ASSOC_VOC       = "Assoc-voc"
    NINTH           = "9th"
    SEVENTH_EIGHTH  = "7th-8th"
    TWELFTH         = "12th"
    MASTERS         = "Masters"
    FIRST_FOURTH    = "1st-4th"
    TENTH           = "10th"
    DOCTORATE       = "Doctorate"
    FIFTH_SIXTH     = "5th-6th"
    PRESCHOOL       = "Preschool"


class MaritalStatusEnum(str, Enum):
    MARRIED_CIV         = "Married-civ-spouse"
    DIVORCED            = "Divorced"
    NEVER_MARRIED       = "Never-married"
    SEPARATED           = "Separated"
    WIDOWED             = "Widowed"
    MARRIED_SPOUSE_ABSENT = "Married-spouse-absent"
    MARRIED_AF          = "Married-AF-spouse"


class OccupationEnum(str, Enum):
    TECH_SUPPORT        = "Tech-support"
    CRAFT_REPAIR        = "Craft-repair"
    OTHER_SERVICE       = "Other-service"
    SALES               = "Sales"
    EXEC_MANAGERIAL     = "Exec-managerial"
    PROF_SPECIALTY      = "Prof-specialty"
    HANDLERS_CLEANERS   = "Handlers-cleaners"
    MACHINE_OP_INSPCT   = "Machine-op-inspct"
    ADM_CLERICAL        = "Adm-clerical"
    FARMING_FISHING     = "Farming-fishing"
    TRANSPORT_MOVING    = "Transport-moving"
    PRIV_HOUSE_SERV     = "Priv-house-serv"
    PROTECTIVE_SERV     = "Protective-serv"
    ARMED_FORCES        = "Armed-Forces"


class RelationshipEnum(str, Enum):
    WIFE                = "Wife"
    OWN_CHILD           = "Own-child"
    HUSBAND             = "Husband"
    NOT_IN_FAMILY       = "Not-in-family"
    OTHER_RELATIVE      = "Other-relative"
    UNMARRIED           = "Unmarried"


class RaceEnum(str, Enum):
    WHITE               = "White"
    ASIAN_PAC_ISLANDER  = "Asian-Pac-Islander"
    AMER_INDIAN_ESKIMO  = "Amer-Indian-Eskimo"
    OTHER               = "Other"
    BLACK               = "Black"


class SexEnum(str, Enum):
    MALE    = "Male"
    FEMALE  = "Female"


class CountryEnum(str, Enum):
    UNITED_STATES   = "United-States"
    CUBA            = "Cuba"
    JAMAICA         = "Jamaica"
    INDIA           = "India"
    MEXICO          = "Mexico"
    SOUTH           = "South"
    JAPAN           = "Japan"
    CHINA           = "China"
    PHILIPPINES     = "Philippines"
    GERMANY         = "Germany"
    PUERTO_RICO     = "Puerto-Rico"
    CANADA          = "Canada"
    EL_SALVADOR     = "El-Salvador"
    INDIA2          = "India"
    OTHER           = "Other"


# ---------------------------------------------------------------------------
# Core input schema — one record sent to /predict
# ---------------------------------------------------------------------------

class PredictionInput(BaseModel):
    """
    A single inference request. Field constraints mirror the UCI dataset's
    actual value ranges — rejecting out-of-range values at the boundary
    prevents model extrapolation from ever reaching the model silently.
    """

    # Numeric features
    age: int = Field(
        ...,
        ge=17, le=90,
        description="Age of the individual. UCI range: 17–90.",
        examples=[35],
    )
    fnlwgt: int = Field(
        ...,
        ge=10_000, le=1_500_000,
        description="Final sampling weight assigned by the Census Bureau.",
        examples=[215_646],
    )
    education_num: int = Field(
        ...,
        ge=1, le=16,
        description="Ordinal encoding of education level (1=Preschool, 16=Doctorate).",
        examples=[9],
    )
    capital_gain: int = Field(
        ...,
        ge=0, le=99_999,
        description="Capital gains recorded in the census year.",
        examples=[0],
    )
    capital_loss: int = Field(
        ...,
        ge=0, le=4_356,
        description="Capital losses recorded in the census year.",
        examples=[0],
    )
    hours_per_week: int = Field(
        ...,
        ge=1, le=99,
        description="Self-reported hours worked per week.",
        examples=[40],
    )

    # Categorical features — Enum-constrained so only known values pass
    workclass: WorkclassEnum = Field(
        ...,
        description="Employment sector.",
        examples=["Private"],
    )
    education: EducationEnum = Field(
        ...,
        description="Highest education level attained.",
        examples=["HS-grad"],
    )
    marital_status: MaritalStatusEnum = Field(
        ...,
        description="Marital status.",
        examples=["Never-married"],
    )
    occupation: OccupationEnum = Field(
        ...,
        description="Occupation category.",
        examples=["Craft-repair"],
    )
    relationship: RelationshipEnum = Field(
        ...,
        description="Relationship to household head.",
        examples=["Not-in-family"],
    )
    race: RaceEnum = Field(
        ...,
        description="Self-identified race.",
        examples=["White"],
    )
    sex: SexEnum = Field(
        ...,
        description="Sex.",
        examples=["Male"],
    )
    native_country: CountryEnum = Field(
        ...,
        description="Country of origin.",
        examples=["United-States"],
    )

    # Cross-field business rule: capital_gain and capital_loss cannot both be
    # non-zero for the same individual in the same year.
    @model_validator(mode="after")
    def capital_gain_and_loss_mutually_exclusive(self) -> "PredictionInput":
        if self.capital_gain > 0 and self.capital_loss > 0:
            raise ValueError(
                "capital_gain and capital_loss cannot both be non-zero. "
                "A person cannot realise gains and losses simultaneously in the same record."
            )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlwgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States",
            }
        }
    }


# ---------------------------------------------------------------------------
# Prediction response — what the API returns to the caller
# ---------------------------------------------------------------------------

class PredictionOutput(BaseModel):
    """
    Response returned from /predict.
    prediction_id ties this response to the row logged in PostgreSQL,
    making it trivially searchable during post-hoc audits.
    """

    prediction_id: UUID = Field(
        default_factory=uuid4,
        description="Unique ID for this prediction — stored in PostgreSQL for audit trail.",
    )
    predicted_label: str = Field(
        ...,
        description="Model output: '>50K' or '<=50K'.",
        examples=[">50K"],
    )
    probability_above_50k: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Calibrated probability that income exceeds $50K.",
        examples=[0.73],
    )
    model_version: str = Field(
        ...,
        description="MLflow model version that served this prediction.",
        examples=["1"],
    )
    served_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of when this prediction was generated.",
    )

    @field_validator("predicted_label")
    @classmethod
    def label_must_be_valid(cls, v: str) -> str:
        allowed = {">50K", "<=50K"}
        if v not in allowed:
            raise ValueError(f"predicted_label must be one of {allowed}, got '{v}'")
        return v


# ---------------------------------------------------------------------------
# Logged prediction record — what gets written to PostgreSQL
# Superset of PredictionOutput: includes the raw input for drift monitoring.
# ---------------------------------------------------------------------------

class PredictionLog(BaseModel):
    """
    Full record persisted to the `predictions` table in PostgreSQL.
    Storing inputs alongside outputs is what enables Evidently to compare
    live traffic distributions against the training baseline.
    """

    prediction_id: UUID
    input_data: PredictionInput
    predicted_label: str
    probability_above_50k: float
    model_version: str
    served_at: datetime
    response_time_ms: Optional[float] = Field(
        default=None,
        description="End-to-end latency in milliseconds for this request.",
    )


# ---------------------------------------------------------------------------
# Health-check response — used by /health endpoint and Docker health checks
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model_version: str = Field(..., examples=["1"])
    model_stage: str = Field(..., examples=["Production"])
    uptime_seconds: float = Field(..., description="Seconds since the API process started.")


# ---------------------------------------------------------------------------
# Error response — consistent error envelope across all 4xx / 5xx responses
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Machine-readable error code.", examples=["validation_error"])
    detail: str = Field(..., description="Human-readable description of what went wrong.")
    request_id: Optional[UUID] = Field(
        default=None,
        description="Echo of prediction_id if the error occurred mid-prediction.",
    )