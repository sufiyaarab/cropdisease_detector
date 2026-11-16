"""Care plan engine.

Rule from Report 2: nothing is shown unless it passes BOTH checks.
  - legal_check:       allowed for home gardeners in Ontario (Allowable List / O. Reg. 63/09)
  - suitability_check: recommended for this problem by OMAFRA guidance

The seed file contains only non-chemical practices, and they are still drafts
(suitability_check is false). While any shown step is a draft, the plan is
marked "draft" and carries a disclaimer. Set SHOW_DRAFT_STEPS=false in
production so unreviewed steps are hidden instead.
"""
import json
import os
from pathlib import Path

from app.pipeline.classes import BY_LABEL
from app.schemas import CarePlan, CareStep

_DATA = json.loads((Path(__file__).parent.parent / "data" / "treatments.json").read_text())

DRAFT_DISCLAIMER = (
    "Draft guidance for development. It has not been reviewed against Ontario (OMAFRA) "
    "recommendations yet, and it is not a substitute for advice from a local expert."
)
REVIEWED_DISCLAIMER = "General guidance for Ontario home gardens. Ask a local expert if the problem keeps spreading."


def build_care_plan(label: str, show_drafts: bool | None = None) -> CarePlan | None:
    entry = _DATA.get(label)
    if entry is None:
        return None
    if show_drafts is None:
        show_drafts = os.getenv("SHOW_DRAFT_STEPS", "true").lower() == "true"

    steps: list[CareStep] = []
    any_draft = False
    for raw in entry["steps"]:
        if not raw["legal_check"]:
            continue  # never shown, draft mode or not
        if not raw["suitability_check"]:
            if not show_drafts:
                continue
            any_draft = True
        steps.append(CareStep(title=raw["title"], detail=raw["detail"], kind=raw["kind"], source=raw["source"]))

    return CarePlan(
        label=label,
        display_name=BY_LABEL[label].display_name,
        summary=entry["summary"],
        steps=steps,
        review_status="draft" if any_draft else "reviewed",
        disclaimer=DRAFT_DISCLAIMER if any_draft else REVIEWED_DISCLAIMER,
    )
