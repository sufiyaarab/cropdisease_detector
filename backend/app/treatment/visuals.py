"""Disease pictures content for the result page (stages, captions, drawing inputs).

The frontend draws the leaves from `look`, so no photo is shipped yet. Real
photos go in only when their licence allows reuse, credited under each photo.
"""
import json
from pathlib import Path

from app.schemas import DiseaseVisuals, LeafLook

_DATA = json.loads((Path(__file__).parent.parent / "data" / "visuals.json").read_text())

SOURCE = "Drawings, not photos. Draft descriptions: check against OMAFRA guidance before release."


def get_visuals(label: str) -> DiseaseVisuals | None:
    entry = _DATA.get(label)
    if entry is None:
        return None
    return DiseaseVisuals(
        label=label,
        look=LeafLook(**entry["look"]),
        examples=entry["examples"],
        stages=entry["stages"],
        source=SOURCE,
    )
