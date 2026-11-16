"""FastAPI entry point. Run locally with:  uvicorn app.main:app --reload"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import errors
from app.api import auth, outbreaks, plants, scans
from app.config import settings
from app.deps import get_model

app = FastAPI(title="Crop Disease Monitoring API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
errors.install(app)
for module in (auth, scans, plants, outbreaks):
    app.include_router(module.router)


@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok", "model_version": get_model().version, "threshold": settings.confidence_threshold}
