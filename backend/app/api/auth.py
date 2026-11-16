"""Sign up, log in, log out.

DEVELOPMENT ONLY. This keeps accounts in memory so the flow works end to end.
The production plan is Supabase Auth: the frontend signs in with Supabase and
the backend only verifies the token in `deps.current_user`.
"""
import re

from fastapi import APIRouter, Depends

from app.deps import bearer_token, get_store
from app.errors import AppError
from app.schemas import Credentials, Session
from app.store import Store

router = APIRouter(prefix="/auth", tags=["auth"])
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")


def _clean_email(raw: str) -> str:
    email = raw.strip().lower()
    if not EMAIL.match(email):
        raise AppError(422, "invalid_email", "Enter a valid email address.")
    return email


@router.post("/signup", response_model=Session, status_code=201)
def signup(body: Credentials, store: Store = Depends(get_store)) -> Session:
    email = _clean_email(body.email)
    user = store.create_user(email, body.password)
    if user is None:
        raise AppError(409, "email_taken", "An account with this email already exists. Try logging in.")
    return Session(token=store.new_token(email), user_id=user["id"], email=email)


@router.post("/login", response_model=Session)
def login(body: Credentials, store: Store = Depends(get_store)) -> Session:
    email = _clean_email(body.email)
    user = store.check_password(email, body.password)
    if user is None:
        raise AppError(401, "wrong_credentials", "Email or password is incorrect.")
    return Session(token=store.new_token(email), user_id=user["id"], email=email)


@router.post("/logout", status_code=204)
def logout(token: str = Depends(bearer_token), store: Store = Depends(get_store)) -> None:
    store.drop_token(token)
