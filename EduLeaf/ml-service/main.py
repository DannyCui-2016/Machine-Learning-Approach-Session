from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from routers import exam_generation
from config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 EduLeaf ML Service starting up…")
    yield
    print("🛑 EduLeaf ML Service shutting down…")

app = FastAPI(
    title="EduLeaf ML Service",
    description="AI-powered exam generation from documents or by difficulty level",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(exam_generation.router, prefix="/api/ml", tags=["Exam Generation"])

@app.get("/health")
async def health():
    return {"status": "ok", "service": "EduLeaf ML Service"}
