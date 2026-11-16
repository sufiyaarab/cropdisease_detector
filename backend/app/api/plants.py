"""My Plants flow: saved plants -> plant page (photo history and results) -> add new photo / updated care plan."""
from fastapi import APIRouter, Depends

from app.deps import current_user, get_store
from app.errors import AppError
from app.schemas import PlantCreate, PlantDetail, PlantSummary
from app.store import Store

router = APIRouter(prefix="/plants", tags=["plants"])


def _summary(store: Store, plant: dict) -> dict:
    scans = store.scans_for_plant(plant["id"])
    diagnosed = [s for s in scans if s.status == "diagnosed"]
    return {
        "id": plant["id"],
        "name": plant["name"],
        "crop": plant["crop"],
        "created_at": plant["created_at"],
        "scan_count": len(scans),
        "latest": diagnosed[0].top if diagnosed else None,
        "latest_scan_at": scans[0].created_at if scans else None,
    }


@router.post("", response_model=PlantSummary, status_code=201)
def save_plant(body: PlantCreate, user: dict = Depends(current_user), store: Store = Depends(get_store)):
    name = body.name.strip()
    if not name:
        raise AppError(422, "name_required", "Give your plant a name.")
    scan = store.get_scan(user["id"], body.scan_id)
    if scan is None:
        raise AppError(404, "scan_not_found", "We could not find that scan.")
    result = scan["result"]
    if result.plant_id is not None:
        raise AppError(409, "already_saved", "This scan is already saved to a plant.")
    plant = store.create_plant(user["id"], name, result.crop)
    result.plant_id = plant["id"]
    return _summary(store, plant)


@router.get("", response_model=list[PlantSummary])
def list_plants(user: dict = Depends(current_user), store: Store = Depends(get_store)):
    return [_summary(store, p) for p in store.plants_for(user["id"])]


@router.get("/{plant_id}", response_model=PlantDetail)
def get_plant(plant_id: str, user: dict = Depends(current_user), store: Store = Depends(get_store)):
    plant = store.get_plant(user["id"], plant_id)
    if plant is None:
        raise AppError(404, "plant_not_found", "We could not find that plant.")
    scans = store.scans_for_plant(plant_id)
    diagnosed = [s for s in scans if s.status == "diagnosed"]
    # The plant's care plan always follows its most recent confident diagnosis.
    return {**_summary(store, plant), "scans": scans, "care_plan": diagnosed[0].care_plan if diagnosed else None}


@router.delete("/{plant_id}", status_code=204)
def delete_plant(plant_id: str, user: dict = Depends(current_user), store: Store = Depends(get_store)) -> None:
    if store.get_plant(user["id"], plant_id) is None:
        raise AppError(404, "plant_not_found", "We could not find that plant.")
    store.delete_plant(plant_id)
