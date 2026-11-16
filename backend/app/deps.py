"""Shared objects the endpoints depend on: the store, the model, the current user."""
from fastapi import Header

from app.config import settings
from app.errors import AppError
from app.pipeline.model import Classifier, StubClassifier
from app.store import Store

store = Store()
if settings.seed_demo_user:
    store.seed_demo_user()
model: Classifier = StubClassifier(version=settings.model_version)


def get_store() -> Store:
    return store


def get_model() -> Classifier:
    return model


def bearer_token(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise AppError(401, "not_signed_in", "Please sign in to continue.")
    return authorization[7:].strip()


def current_user(authorization: str | None = Header(default=None)) -> dict:
    user = store.user_for_token(bearer_token(authorization))
    if user is None:
        raise AppError(401, "session_expired", "Your session has ended. Please sign in again.")
    return user
