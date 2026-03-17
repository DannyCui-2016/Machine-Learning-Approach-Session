"""
File Parser Service
Extracts plain text from PDF, DOCX, and image files.
"""

import os
import logging

logger = logging.getLogger(__name__)


def parse_file(path: str, suffix: str) -> str:
    """Route file to the correct parser based on extension."""
    suffix = suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(path)
    elif suffix in (".docx",):
        return _parse_docx(path)
    elif suffix in (".jpg", ".jpeg", ".png", ".webp"):
        return _parse_image(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _parse_pdf(path: str) -> str:
    try:
        import PyPDF2
        text = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text).strip()
    except Exception as e:
        logger.error(f"PDF parse error: {e}")
        return ""


def _parse_docx(path: str) -> str:
    try:
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        logger.error(f"DOCX parse error: {e}")
        return ""


def _parse_image(path: str) -> str:
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(path)
        return pytesseract.image_to_string(img).strip()
    except Exception as e:
        logger.error(f"Image OCR error: {e}")
        return ""
