"""
UDMS Image Classifier — FastAPI Application.
"""

import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.dependencies import load_classifier
from app.middleware import RequestLoggingMiddleware
from app.routes import classify, health

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup; nothing special on shutdown."""
    load_classifier()
    logging.getLogger(__name__).info("Classifier loaded successfully")
    yield


app = FastAPI(
    title="UDMS Image Classifier",
    description="Urban Disorder Monitoring System — classifies civic-issue "
    "images into 7 categories using a TFLite MobileNetV2 model.",
    version=settings.MODEL_VERSION,
    lifespan=lifespan,
)

# --- middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

# --- routes ---
app.include_router(health.router)
app.include_router(classify.router)


@app.get("/")
def root() -> dict:
    """Welcome endpoint."""
    return {
        "service": "UDMS Image Classifier",
        "version": settings.MODEL_VERSION,
        "docs": "/docs",
    }

