"""
ML Service – Exam Generation Router
Handles two modes:
  1. generate-from-file: parse uploaded document → extract text → generate questions
  2. generate-auto: generate questions by subject + difficulty level
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import Optional
import tempfile, os

from services.file_parser import parse_file
from services.exam_generator import generate_from_text, generate_auto

router = APIRouter()


class AutoGenerateRequest(BaseModel):
    subject: str
    level: str


# ── POST /api/ml/generate-from-file ──────────────────────────────────────────
@router.post("/generate-from-file")
async def generate_from_file_endpoint(
    file: UploadFile = File(...),
    subject: str = Form(...),
):
    if not file:
        raise HTTPException(status_code=400, detail="File is required")

    suffix = os.path.splitext(file.filename or "upload")[1].lower()
    allowed = {".pdf", ".docx", ".jpg", ".jpeg", ".png", ".webp"}
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        text = parse_file(tmp_path, suffix)
        if not text or len(text.strip()) < 50:
            raise HTTPException(status_code=422, detail="Could not extract enough text from the file")

        exam = generate_from_text(text, subject)
        return exam
    finally:
        os.unlink(tmp_path)


# ── POST /api/ml/generate-auto ────────────────────────────────────────────────
@router.post("/generate-auto")
async def generate_auto_endpoint(body: AutoGenerateRequest):
    VALID_SUBJECTS = {"spanish", "german", "english-pte"}
    if body.subject not in VALID_SUBJECTS:
        raise HTTPException(status_code=400, detail=f"Unknown subject: {body.subject}")

    exam = generate_auto(body.subject, body.level)
    return exam
