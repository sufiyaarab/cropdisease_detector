"""Shapes of everything the API sends and receives.

FastAPI turns these into contracts/openapi.json, which the frontend types mirror.
Change a shape here only together with frontend/src/api/types.ts.
"""
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

Crop = Literal["tomato", "apple", "corn"]
Category = Literal["disease", "pest", "healthy"]
ScanStatus = Literal["diagnosed", "not_sure", "rejected"]


class ApiError(BaseModel):
    code: str
    message: str


class Credentials(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(min_length=8, max_length=128)


class Session(BaseModel):
    token: str
    user_id: str
    email: str


class PhotoIssue(BaseModel):
    """One reason a photo was not accepted, written for the gardener."""

    photo_index: int
    code: str
    message: str
    tip: str


class Candidate(BaseModel):
    label: str
    display_name: str
    category: Category
    confidence: float


class CareStep(BaseModel):
    title: str
    detail: str
    kind: Literal["practice", "product"]
    source: str


class CarePlan(BaseModel):
    label: str
    display_name: str
    summary: str
    steps: list[CareStep]
    review_status: Literal["draft", "reviewed"]
    disclaimer: str


class ScanResult(BaseModel):
    id: str
    status: ScanStatus
    crop: Crop
    created_at: datetime
    photo_count: int
    issues: list[PhotoIssue] = []
    top: Candidate | None = None
    candidates: list[Candidate] = []
    care_plan: CarePlan | None = None
    model_version: str | None = None
    plant_id: str | None = None
    # Set when this scan belongs to a saved plant and its diagnosis differs
    # from the plant's previous diagnosed scan.
    changed_from: str | None = None


class PlantCreate(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    scan_id: str


class PlantSummary(BaseModel):
    id: str
    name: str
    crop: Crop
    created_at: datetime
    scan_count: int
    latest: Candidate | None = None
    latest_scan_at: datetime | None = None


class PlantDetail(PlantSummary):
    care_plan: CarePlan | None = None
    scans: list[ScanResult] = []


class OutbreakAlert(BaseModel):
    id: str
    crop: Crop
    label: str
    display_name: str
    postal_prefix: str
    reported_on: date
    report_count: int
    note: str


class OutbreakResponse(BaseModel):
    postal_prefix: str
    alerts: list[OutbreakAlert]
    is_sample_data: bool
