"""In-memory data store for local development.

Everything is lost when the server restarts. The method names match the tables
in db/migrations/001_init.sql, so a Supabase-backed class with the same
methods can replace this one later without changing the API code.
"""
import hashlib
import secrets
import uuid
from datetime import datetime, timezone

from app.schemas import ScanResult


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)


# Ready-made account for local testing, because accounts are wiped on every restart.
# The same values are in frontend/src/pages/LoginPage.tsx.
DEMO_EMAIL = "demo@example.com"
DEMO_PASSWORD = "demo-password"


class Store:
    def __init__(self) -> None:
        self.users: dict[str, dict] = {}  # email -> user
        self.tokens: dict[str, str] = {}  # token -> email
        self.scans: dict[str, dict] = {}  # scan id -> {owner, result, photos}
        self.plants: dict[str, dict] = {}  # plant id -> plant

    def seed_demo_user(self) -> None:
        self.create_user(DEMO_EMAIL, DEMO_PASSWORD)

    # users ---------------------------------------------------------------
    def create_user(self, email: str, password: str) -> dict | None:
        if email in self.users:
            return None
        salt = secrets.token_bytes(16)
        user = {"id": str(uuid.uuid4()), "email": email, "salt": salt, "hash": _hash(password, salt)}
        self.users[email] = user
        return user

    def check_password(self, email: str, password: str) -> dict | None:
        user = self.users.get(email)
        if user and secrets.compare_digest(user["hash"], _hash(password, user["salt"])):
            return user
        return None

    def new_token(self, email: str) -> str:
        token = secrets.token_urlsafe(32)
        self.tokens[token] = email
        return token

    def user_for_token(self, token: str) -> dict | None:
        return self.users.get(self.tokens.get(token, ""))

    def drop_token(self, token: str) -> None:
        self.tokens.pop(token, None)

    # scans ---------------------------------------------------------------
    def save_scan(self, owner_id: str, result: ScanResult, photos: list[bytes]) -> None:
        self.scans[result.id] = {"owner": owner_id, "result": result, "photos": photos}

    def get_scan(self, owner_id: str, scan_id: str) -> dict | None:
        scan = self.scans.get(scan_id)
        return scan if scan and scan["owner"] == owner_id else None

    def scans_for_plant(self, plant_id: str) -> list[ScanResult]:
        found = [s["result"] for s in self.scans.values() if s["result"].plant_id == plant_id]
        return sorted(found, key=lambda r: r.created_at, reverse=True)

    # plants --------------------------------------------------------------
    def create_plant(self, owner_id: str, name: str, crop: str) -> dict:
        plant = {"id": str(uuid.uuid4()), "owner": owner_id, "name": name, "crop": crop, "created_at": _now()}
        self.plants[plant["id"]] = plant
        return plant

    def get_plant(self, owner_id: str, plant_id: str) -> dict | None:
        plant = self.plants.get(plant_id)
        return plant if plant and plant["owner"] == owner_id else None

    def plants_for(self, owner_id: str) -> list[dict]:
        mine = [p for p in self.plants.values() if p["owner"] == owner_id]
        return sorted(mine, key=lambda p: p["created_at"], reverse=True)

    def delete_plant(self, plant_id: str) -> None:
        self.plants.pop(plant_id, None)
        for scan in self.scans.values():
            if scan["result"].plant_id == plant_id:
                scan["result"].plant_id = None
