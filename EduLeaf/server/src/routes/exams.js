const express = require('express');
const router  = express.Router();
const multer  = require('multer');
const path    = require('path');
const { v4: uuidv4 } = require('uuid');
const axios   = require('axios');
const { Exam, ExamRecord } = require('../models/models');

// ── Multer config ─────────────────────────────────────────────────────────────
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, path.join(__dirname, '../../uploads')),
  filename:    (req, file, cb) => cb(null, `${uuidv4()}-${file.originalname}`),
});
const upload = multer({
  storage,
  limits: { fileSize: 20 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowed = ['application/pdf',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'image/jpeg', 'image/png', 'image/webp'];
    cb(null, allowed.includes(file.mimetype));
  },
});

const ML_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

// ── POST /api/exams/generate-from-file ───────────────────────────────────────
router.post('/generate-from-file', upload.single('file'), async (req, res, next) => {
    console.log('✅ /generate-from-file route hit');
    console.log('File received:', req.file);
    console.log('Body received:', req.body);
  try {
    const { subject } = req.body;
    const file = req.file;
    if (!file) return res.status(400).json({ error: 'File is required' });

    let textContent = '';
    const fs = require('fs');
    if (file.mimetype === 'application/pdf') {
      const pdfParse = require('pdf-parse');
      const dataBuffer = fs.readFileSync(file.path);
      const data = await pdfParse(dataBuffer);
      textContent = data.text;
    } else {
      // Basic fallback for other file types - for simplicity in this session we focus on PDF
      textContent = 'File content not extracted. Please upload a PDF.';
    }

    if (!textContent || textContent.trim().length < 50) {
      return res.status(422).json({ error: 'Could not extract enough text from the file. Please ensure it is a text-based PDF.' });
    }
    // const Anthropic = require('@anthropic-ai/sdk');
    // const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
    const { GoogleGenerativeAI } = require('@google/generative-ai');
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

    const prompt = `
    You are an expert exam generator. Create a structured exam based *strictly* on the following text extracted from a document. The subject is "${subject}".

    Extracted Text:
    ---
    ${textContent.substring(0, 4000)}
    ---

    Generate an exam with these sections:
    - multipleChoice: 5 questions (options: A, B, C, D), 4 points each.
    - fillIn: 3 fill-in-the-blank questions (provide the exactly 1-2 word answer), 4 points each.
    - reading: 2 reading comprehension questions. Write a "passage" of 2-3 sentences taken from the text. Do NOT include the answer in the passage. The "answer" field should be a short phrase (2-5 words) that answers the question but does NOT appear verbatim in the passage. 10 points each.
    - writing: 1 writing prompt related to the text, 20 points.

    Respond ONLY with valid JSON in this exact structure, with no markdown formatting around it:
    {
      "title": "Exam based on Document",
      "sections": {
        "multipleChoice": [
          { "id": "mc1", "type": "mc", "question": "...", "options": ["...", "...", "...", "..."], "answer": "A", "points": 4 }
        ],
        "fillIn": [
          { "id": "fi1", "type": "fill", "question": "...", "answer": "...", "points": 4 }
        ],
        "reading": [
          { "id": "r1", "type": "reading", "passage": "...", "question": "...", "answer": "...", "points": 10 }
        ],
        "writing": [
          { "id": "w1", "type": "writing", "question": "...", "rubric": "...", "minWords": 50, "answer": "", "points": 20 }
        ]
      }
    }
    `;

    const result = await model.generateContent(prompt);
    const mlResult = JSON.parse(result.response.text());
    // Persist exam
    const exam = await Exam.create({
      title:       mlResult.title || `${subject} Exam – File`,
      subject,
      source:      'file',
      sectionsJson: JSON.stringify(mlResult.sections),
    });

    res.json({ id: exam.id, title: exam.title, subject, sections: mlResult.sections });
  } catch (err) {
    console.error('❌ Error in generate-from-file:', err);
    console.error('Full error:', err.stack); 
    next(err);
  }
});

// ── POST /api/exams/generate-auto ─────────────────────────────────────────────
router.post('/generate-auto', async (req, res, next) => {
  try {
    const { subject, level } = req.body;
    if (!subject || !level) return res.status(400).json({ error: 'subject and level required' });

    const { data: mlResult } = await axios.post(`${ML_URL}/api/ml/generate-auto`, { subject, level }, { timeout: 30000 });

    const exam = await Exam.create({
      title:       mlResult.title || `${subject} – ${level}`,
      subject,
      level,
      source:      'auto',
      sectionsJson: JSON.stringify(mlResult.sections),
    });

    res.json({ id: exam.id, title: exam.title, subject, level, sections: mlResult.sections });
  } catch (err) {
    console.error("Error in generate-auto:", err);
    next(err);
  }
});

// ── GET /api/exams/history ────────────────────────────────────────────────────
router.get('/history', async (req, res, next) => {
  try {
    const records = await ExamRecord.findAll({
      include: [{ association: 'exam', attributes: ['title', 'subject', 'level'] }],
      order: [['createdAt', 'DESC']],
      limit: 20,
    });
    res.json(records);
  } catch (err) { next(err); }
});

// ── GET /api/exams/:id ────────────────────────────────────────────────────────
router.get('/:id', async (req, res, next) => {
  try {
    const exam = await Exam.findByPk(req.params.id);
    if (!exam) return res.status(404).json({ error: 'Exam not found' });
    res.json({ id: exam.id, title: exam.title, subject: exam.subject, level: exam.level, sections: exam.sections });
  } catch (err) { next(err); }
});

// ── POST /api/exams/:id/submit ────────────────────────────────────────────────
router.post('/:id/submit', async (req, res, next) => {
  try {
    const { answers } = req.body;
    const exam = await Exam.findByPk(req.params.id);
    if (!exam) return res.status(404).json({ error: 'Exam not found' });

    const sections = exam.sections;
    let earned = 0, total = 0;
    Object.values(sections).forEach((qs) => {
      qs.forEach((q) => {
        total += q.points || 0;
        const ua = (answers[q.id] || '').toString().trim().toLowerCase();
        const ca = (q.answer || '').toString().trim().toLowerCase();
        if (q.type === 'writing') {
          if (ua.split(' ').length >= (q.minWords || 50) * 0.5) earned += (q.points || 0) * 0.6;
        } else if (q.type === 'mc') {
          if (ua === ca) earned += q.points || 0;
        } else {
          if (ua && (ua.includes(ca) || ca.includes(ua))) earned += q.points || 0;
        }
      });
    });

    const score = Math.round((earned / total) * 100);
    await ExamRecord.create({ examId: exam.id, score, total: 100, answersJson: JSON.stringify(answers) });
    res.json({ score, total: 100, message: 'Submitted successfully!' });
  } catch (err) { next(err); }
});

// ── POST /api/exams/:id/verify-section ───────────────────────────────────────
router.post('/:id/verify-section', async (req, res, next) => {
  try {
    const { section, answers } = req.body;
    const exam = await Exam.findByPk(req.params.id);
    if (!exam) return res.status(404).json({ error: 'Exam not found' });

    const qs = exam.sections[section] || [];
    let correct = 0;
    const details = {};
    qs.forEach((q) => {
      const ua = (answers[q.id] || '').toString().trim().toLowerCase();
      const ca = (q.answer || '').toString().trim().toLowerCase();
      const ok = q.type === 'mc' ? ua === ca : (ua.includes(ca) || ca.includes(ua));
      details[q.id] = ok;
      if (ok) correct++;
    });
    res.json({ correct, total: qs.length, details });
  } catch (err) { next(err); }
});

// ── Mock helper ───────────────────────────────────────────────────────────────
function getMockExam(subject, level) {
  return {
    id: `mock-${Date.now()}`,
    title: `${subject} Exam – ${level}`,
    subject, level,
    sections: {
      multipleChoice: [
        { id: 'mc1', type: 'mc', question: 'Which word means "Good morning" in Spanish?', options: ['Buenos días','Buenas noches','Hola','Adiós'], answer: 'A', points: 4 },
        { id: 'mc2', type: 'mc', question: 'What is the plural of "el libro"?', options: ['los libros','las libros','el libros','los libro'], answer: 'A', points: 4 },
      ],
      fillIn: [
        { id: 'fi1', type: 'fill', question: 'Complete: "Yo _____ estudiante." (to be)', answer: 'soy', points: 4 },
      ],
      listening: [
        { id: 'li1', type: 'listening', audioText: 'Me llamo Ana. Tengo quince años y vivo en Madrid.', question: "What is the speaker's name and age?", answer: 'Ana, 15 years old', points: 5 },
      ],
      reading: [
        { id: 'ri1', type: 'reading', passage: 'La familia García vive en Barcelona. Tienen tres hijos.', question: 'Where does the García family live?', answer: 'Barcelona', points: 10 },
      ],
      writing: [
        { id: 'wi1', type: 'writing', question: 'Write 50–80 words about your school day.', rubric: 'Vocabulary (5), Grammar (5), Coherence (5), Word count (5)', minWords: 50, answer: '', points: 20 },
      ],
    },
  };
}

module.exports = router;
