# 🍃 EduLeaf - Educational Platform

A bilingual (English/Chinese) educational website for students (primary to university) and parents.

## Architecture

| Service | Tech | Port |
|---------|------|------|
| Frontend | Next.js 14 (App Router) | 3000 |
| Backend | Node.js + Express | 3001 |
| ML Service | Python + FastAPI | 8000 |

## Quick Start

```bash
# 1. Frontend
cd client && npm install && npm run dev

# 2. Backend
cd server && npm install && npm run dev

# 3. ML Service
cd ml-service && pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Modules

1. **Exam Design** (Module 1) — AI-powered exam generation for Spanish, German, English PTE
2. **Math Module** (Module 2) — NZ curriculum-aligned math exercises (Year 1–13 template)
