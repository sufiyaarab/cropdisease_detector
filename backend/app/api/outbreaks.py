"""Local outbreaks flow: enter postal code -> view alerts.

Alerts come from a sample file for now. The real version builds them from
opt-in scans (label + date + postal prefix only, never GPS) after admin review.
"""
import json
import re
from pathlib import Path

from fastapi import APIRouter, Depends, Query

from app.deps import current_user
from app.errors import AppError
from app.pipeline.classes import BY_LABEL
from app.schemas import OutbreakAlert, OutbreakResponse

router = APIRouter(prefix="/outbreaks", tags=["outbreaks"])
_ALERTS = json.loads((Path(__file__).parent.parent / "data" / "outbreaks.json").read_text())

# First three characters of a Canadian postal code (the "forward sortation area").
FSA = re.compile(r"^[ABCEGHJ-NPRSTVXY][0-9][ABCEGHJ-NPRSTV-Z]")
ONTARIO_LETTERS = "KLMNP"


def postal_prefix(raw: str) -> str:
    """Accepts 'L1H', 'l1h 7k4' or 'L1H7K4' and returns 'L1H'. Only the prefix is ever used."""
    cleaned = re.sub(r"[\s-]", "", raw).upper()
    if not FSA.match(cleaned) or len(cleaned) not in (3, 6):
        raise AppError(422, "invalid_postal_code", "Enter a Canadian postal code, like L1H or L1H 7K4.")
    if cleaned[0] not in ONTARIO_LETTERS:
        raise AppError(422, "outside_ontario", "Outbreak alerts only cover Ontario postal codes for now.")
    return cleaned[:3]


@router.get("", response_model=OutbreakResponse)
def list_outbreaks(postal_code: str = Query(max_length=10), _: dict = Depends(current_user)) -> OutbreakResponse:
    prefix = postal_prefix(postal_code)
    alerts = [
        OutbreakAlert(**a, display_name=BY_LABEL[a["label"]].display_name)
        for a in _ALERTS
        if a["postal_prefix"] == prefix
    ]
    alerts.sort(key=lambda a: a.reported_on, reverse=True)
    return OutbreakResponse(postal_prefix=prefix, alerts=alerts, is_sample_data=True)
