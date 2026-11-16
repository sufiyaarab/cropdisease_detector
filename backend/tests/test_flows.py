"""Walks the three flows from the user-flow diagram end to end."""
from tests.conftest import make_photo, scan


# --- Login / sign up -----------------------------------------------------
def test_signup_login_logout(client):
    creds = {"email": "New@Example.com ", "password": "long-enough"}
    assert client.post("/auth/signup", json=creds).status_code == 201
    assert client.post("/auth/signup", json=creds).json()["code"] == "email_taken"
    token = client.post("/auth/login", json=creds).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/plants", headers=headers).status_code == 200
    assert client.post("/auth/logout", headers=headers).status_code == 204
    assert client.get("/plants", headers=headers).json()["code"] == "session_expired"


def test_auth_validation(client):
    assert client.post("/auth/signup", json={"email": "nope", "password": "long-enough"}).json()["code"] == "invalid_email"
    assert client.post("/auth/signup", json={"email": "a@b.co", "password": "short"}).json()["code"] == "invalid_request"
    assert client.post("/auth/login", json={"email": "a@b.co", "password": "wrong-pass"}).json()["code"] == "wrong_credentials"
    assert client.get("/plants").json()["code"] == "not_signed_in"


# --- Disease detection flow ----------------------------------------------
def test_scan_gives_diagnosis_confidence_and_care_plan(client, auth):
    body = scan(client, auth).json()
    assert body["status"] == "diagnosed"
    assert body["top"]["label"] == "tomato_early_blight"
    assert body["top"]["confidence"] == 0.9
    assert len(body["candidates"]) == 3
    assert body["care_plan"]["steps"]
    assert body["care_plan"]["review_status"] == "draft" and body["care_plan"]["disclaimer"]
    assert client.get(f"/scans/{body['id']}", headers=auth).json() == body
    photo = client.get(f"/scans/{body['id']}/photos/0", headers=auth)
    assert photo.headers["content-type"] == "image/jpeg"


def test_low_confidence_is_not_sure_with_ranked_candidates_and_no_care_plan(client, auth, model):
    model.probs = [0.10, 0.45, 0.30, 0.10, 0.05]
    body = scan(client, auth).json()
    assert body["status"] == "not_sure"
    assert body["care_plan"] is None
    assert [c["confidence"] for c in body["candidates"]] == [0.45, 0.30, 0.10]


def test_bad_photo_is_rejected_with_a_retake_tip(client, auth):
    body = scan(client, auth, kind="blurry").json()
    assert body["status"] == "rejected"
    assert body["issues"][0]["code"] == "too_blurry" and body["issues"][0]["tip"]
    assert body["top"] is None
    assert client.get(f"/scans/{body['id']}", headers=auth).status_code == 404


def test_scan_request_validation(client, auth):
    assert scan(client, auth, crop="banana").json()["code"] == "unsupported_crop"
    four = [("photos", (f"{i}.jpg", make_photo(seed=i), "image/jpeg")) for i in range(4)]
    assert client.post("/scans", data={"crop": "tomato"}, files=four, headers=auth).json()["code"] == "wrong_photo_count"
    assert client.post("/scans", data={"crop": "tomato"}, headers=auth).json()["code"] == "invalid_request"
    assert scan(client, {}).status_code == 401


def test_multiple_photos_report_which_one_failed(client, auth):
    files = [
        ("photos", ("a.jpg", make_photo("leaf"), "image/jpeg")),
        ("photos", ("b.jpg", make_photo("dark"), "image/jpeg")),
    ]
    body = client.post("/scans", data={"crop": "corn"}, files=files, headers=auth).json()
    assert body["status"] == "rejected"
    assert {i["photo_index"] for i in body["issues"]} == {1}


# --- My Plants flow --------------------------------------------------------
def test_save_plant_then_add_photo_updates_care_plan(client, auth, model):
    first = scan(client, auth).json()
    plant = client.post("/plants", json={"name": " Balcony tomato ", "scan_id": first["id"]}, headers=auth).json()
    assert plant["name"] == "Balcony tomato" and plant["scan_count"] == 1
    assert client.post("/plants", json={"name": "Again", "scan_id": first["id"]}, headers=auth).json()["code"] == "already_saved"

    model.probs = [0.02, 0.03, 0.90, 0.03, 0.02]  # condition changes to late blight
    second = scan(client, auth, plant_id=plant["id"], seed=1).json()
    assert second["plant_id"] == plant["id"]
    assert second["changed_from"] == "Early blight"

    detail = client.get(f"/plants/{plant['id']}", headers=auth).json()
    assert [s["id"] for s in detail["scans"]] == [second["id"], first["id"]]
    assert detail["care_plan"]["label"] == "tomato_late_blight"
    assert client.get("/plants", headers=auth).json()[0]["latest"]["label"] == "tomato_late_blight"


def test_plant_rules(client, auth):
    first = scan(client, auth).json()
    plant = client.post("/plants", json={"name": "Tom", "scan_id": first["id"]}, headers=auth).json()
    assert scan(client, auth, crop="apple", plant_id=plant["id"]).json()["code"] == "crop_mismatch"
    assert scan(client, auth, plant_id="missing").json()["code"] == "plant_not_found"
    assert client.post("/plants", json={"name": "   ", "scan_id": first["id"]}, headers=auth).status_code == 422

    other = client.post("/auth/signup", json={"email": "other@example.com", "password": "long-enough"}).json()
    other_auth = {"Authorization": f"Bearer {other['token']}"}
    assert client.get(f"/plants/{plant['id']}", headers=other_auth).status_code == 404
    assert client.get(f"/scans/{first['id']}", headers=other_auth).status_code == 404

    assert client.delete(f"/plants/{plant['id']}", headers=auth).status_code == 204
    assert client.get("/plants", headers=auth).json() == []


# --- Local outbreaks flow --------------------------------------------------
def test_outbreaks_by_postal_code(client, auth):
    body = client.get("/outbreaks", params={"postal_code": "l1h 7k4"}, headers=auth).json()
    assert body["postal_prefix"] == "L1H" and body["is_sample_data"]
    assert len(body["alerts"]) == 2
    assert body["alerts"][0]["reported_on"] >= body["alerts"][1]["reported_on"]
    assert client.get("/outbreaks", params={"postal_code": "P0T"}, headers=auth).json()["alerts"] == []


def test_outbreak_postal_code_validation(client, auth):
    for bad in ["12345", "L1", "L1H 7K", "D1A"]:
        assert client.get("/outbreaks", params={"postal_code": bad}, headers=auth).json()["code"] == "invalid_postal_code"
    assert client.get("/outbreaks", params={"postal_code": "V6B"}, headers=auth).json()["code"] == "outside_ontario"


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"
