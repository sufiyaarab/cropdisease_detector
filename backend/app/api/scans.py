"""Disease detection flow: Scan plant -> Validate photo and crop -> AI analysis -> Results -> Care plan."""
import uuid
from datetime import datetime, timezone
from typing import get_args

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import Response

from app.config import settings
from app.deps import current_user, get_model, get_store
from app.errors import AppError
from app.pipeline.classes import CLASSES
from app.pipeline.diagnose import diagnose
from app.pipeline.explain import occlusion_map
from app.pipeline.model import Classifier
from app.pipeline.severity import estimate, level_for
from app.pipeline.validate import check_photo, to_stored_jpeg
from app.schemas import Attention, Crop, PhotoIssue, ScanResult, Severity, SpotBox
from app.store import Store
from app.treatment.engine import build_care_plan
from app.treatment.visuals import get_visuals

router = APIRouter(prefix="/scans", tags=["scans"])
CROPS = get_args(Crop)


@router.post("", response_model=ScanResult, status_code=201)
async def create_scan(
    crop: str = Form(...),
    photos: list[UploadFile] = File(...),
    plant_id: str | None = Form(default=None),
    user: dict = Depends(current_user),
    store: Store = Depends(get_store),
    model: Classifier = Depends(get_model),
) -> ScanResult:
    # 1. Validate the request itself.
    if crop not in CROPS:
        raise AppError(422, "unsupported_crop", f"We support {', '.join(CROPS)} for now.")
    if not 1 <= len(photos) <= settings.max_photos:
        raise AppError(422, "wrong_photo_count", f"Send between 1 and {settings.max_photos} photos.")
    if plant_id is not None:
        plant = store.get_plant(user["id"], plant_id)
        if plant is None:
            raise AppError(404, "plant_not_found", "That saved plant no longer exists.")
        if plant["crop"] != crop:
            raise AppError(422, "crop_mismatch", f"This plant is saved as {plant['crop']}.")

    # 2. Validate every photo. One bad photo rejects the scan so the user can retake it.
    images, issues = [], []
    for index, upload in enumerate(photos):
        data = await upload.read(settings.max_upload_bytes + 1)
        checked = check_photo(data, settings)
        if checked.ok:
            images.append(checked.image)
        issues += [PhotoIssue(photo_index=index, code=i.code, message=i.message, tip=i.tip) for i in checked.issues]

    result = ScanResult(
        id=str(uuid.uuid4()),
        status="rejected",
        crop=crop,
        created_at=datetime.now(timezone.utc),
        photo_count=len(photos),
        issues=issues,
    )
    if issues:
        # Rejected scans are returned but not stored: nothing useful to keep.
        return result

    # 3. AI analysis, then the confidence threshold.
    found = diagnose(images, crop, model, settings.confidence_threshold)
    result.status = found.status
    result.top = found.top
    result.candidates = found.candidates
    result.model_version = model.version

    # 4. Care plan, pictures and severity only when the model is confident enough.
    if found.status == "diagnosed":
        result.care_plan = build_care_plan(found.top.label)
        result.visuals = get_visuals(found.top.label)
        result.severity = estimate_severity(images, has_disease=found.top.category != "healthy")
        if settings.show_attention and found.top.category != "healthy":
            index = result.severity.photo_index
            class_index = [c.label for c in CLASSES[crop]].index(found.top.label)
            grid = occlusion_map(model, images[index], crop, class_index)
            result.attention = Attention(photo_index=index, grid=grid)

    # 5. Link to a saved plant and flag a change in condition.
    if plant_id is not None:
        result.plant_id = plant_id
        earlier = [s for s in store.scans_for_plant(plant_id) if s.status == "diagnosed"]
        if earlier and found.status == "diagnosed" and earlier[0].top.label != found.top.label:
            result.changed_from = earlier[0].top.display_name

    store.save_scan(user["id"], result, [to_stored_jpeg(img) for img in images])
    return result


def estimate_severity(images: list, has_disease: bool) -> Severity:
    """Uses the photo with the most spots, so a problem is not played down."""
    estimates = [estimate(img) for img in images]
    worst = max(range(len(estimates)), key=lambda i: estimates[i].leaf_pct)
    found = estimates[worst]
    return Severity(
        leaf_pct=found.leaf_pct,
        level=level_for(found.leaf_pct, has_disease),
        photo_index=worst,
        spots=[SpotBox(x=round(s.x, 4), y=round(s.y, 4), w=round(s.w, 4), h=round(s.h, 4)) for s in found.spots],
    )


@router.get("/{scan_id}", response_model=ScanResult)
def get_scan(scan_id: str, user: dict = Depends(current_user), store: Store = Depends(get_store)) -> ScanResult:
    scan = store.get_scan(user["id"], scan_id)
    if scan is None:
        raise AppError(404, "scan_not_found", "We could not find that scan.")
    return scan["result"]


@router.get("/{scan_id}/photos/{index}", responses={200: {"content": {"image/jpeg": {}}}}, response_class=Response)
def get_photo(
    scan_id: str, index: int, user: dict = Depends(current_user), store: Store = Depends(get_store)
) -> Response:
    scan = store.get_scan(user["id"], scan_id)
    if scan is None or not 0 <= index < len(scan["photos"]):
        raise AppError(404, "photo_not_found", "We could not find that photo.")
    return Response(scan["photos"][index], media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})
