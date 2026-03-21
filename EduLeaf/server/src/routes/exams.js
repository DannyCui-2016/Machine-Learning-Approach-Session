const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const { v4: uuidv4 } = require('uuid');
const axios = require('axios');
const { Exam, ExamRecord } = require('../models/models');

const { createClient } = require('@supabase/supabase-js');
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

// ── Multer config ─────────────────────────────────────────────────────────────
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, path.join(__dirname, '../../uploads')),
  filename: (req, file, cb) => cb(null, `${uuidv4()}-${file.originalname}`),
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

      // Upload to Supabase Storage
      const fileBuffer = fs.readFileSync(file.path);
      const fileName = `${Date.now()}-${file.originalname}`;
      const bucketName = `${subject}uploads`.toLowerCase(); // spanishuploads, germanuploads etc

      await supabase.storage
        .from(bucketName)
        .upload(fileName, fileBuffer, { contentType: file.mimetype });

      // Delete local temporary file
      fs.unlinkSync(file.path);
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
    You are an expert ${subject} language exam generator. Create a structured exam to test the student's ${subject} language knowledge based on the vocabulary, grammar, and phrases found in the following text.

    STRICT RULES:
    - Questions must test ${subject} language knowledge ONLY (vocabulary, grammar, translation, comprehension).
    - Do NOT ask questions about the document itself, file names, titles, lesson numbers, or document structure.
    - Do NOT ask "what does the text start with" or "what is the title of" type questions.
    - Do NOT reference meta-information like "Level 1, Lesson 1" or document formatting.
    - Every question must test real language ability that a student would encounter in a ${subject} exam.
    - Base questions on the ${subject} words, phrases, grammar patterns, and sentences found in the text.

    Extracted Text:
    ---
    ${textContent.substring(0, 4000)}
    ---

    Generate an exam with exactly these 4 sections totaling 100 points:
    1. multipleChoice: 10 questions (options: A, B, C, D), 4 points each (Total 40).
    2. fillIn: 10 fill-in-the-blank questions. Each question must have ONE and ONLY ONE correct answer. Rules:
      - The blank replaces a SPECIFIC word from the original text
      - Include English context in parentheses to make the expected answer unambiguous. Example: "Yo _____ (I am) estudiante." or "La Coca-Cola es para _____ (me/I want it)."
      - The question must clearly indicate what word is expected
      - Do NOT create questions where "ti", "mí", "él" or other pronouns could all be valid
      - If filling a pronoun, specify the subject clearly: "According to Peter, the Coca-Cola is for _____ (Peter himself)"
   3 points each (Total 30).    3. trueFalse: 10 true or false questions based on the text. The answer must be "True" or "False", 2 points each (Total 20).
    4. translation: 5 short sentences from the text to translate from ${subject} into English. The "question" field must be a sentence in ${subject}. The "answer" field must be the English translation ONLY — no ${subject} words mixed in, pure English answer. 2 points each (Total 10).

    Respond ONLY with valid JSON in this exact structure, with no markdown formatting around it:
    {
      "title": "Exam based on Document",
      "sections": {
        "multipleChoice": [
          { "id": "mc1", "type": "mc", "question": "...", "options": ["...", "...", "...", "..."], "answer": "A", "points": 4 }
        ],
        "fillIn": [
          { "id": "fi1", "type": "fill", "question": "...", "answer": "...", "points": 3 }
        ],
        "trueFalse": [
          { "id": "tf1", "type": "tf", "question": "...", "answer": "True", "points": 2 }
        ],
        "translation": [
          { "id": "tr1", "type": "translation", "question": "...", "answer": "...", "points": 2 }
        ]
      }
    }
    `;

    const result = await model.generateContent(prompt);
    const mlResult = JSON.parse(result.response.text());
    const examId = uuidv4();
    Object.entries(mlResult.sections).forEach(([sectionKey, questions]) => {
      questions.forEach((q, idx) => {
        q.id = `${examId.slice(0, 8)}-${sectionKey}-${idx}`;
      });
    });
    // Persist exam
    const originalFileName = file.originalname ? file.originalname.replace(/\.[^/.]+$/, "") : null;
    const finalTitle = originalFileName || `${subject} Exam – File`;

    const exam = await Exam.create({
      id: examId,
      title: finalTitle,
      subject,
      source: 'file',
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
      title: mlResult.title || `${subject} – ${level}`,
      subject,
      level,
      source: 'auto',
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
    const { subject } = req.query;
    const examInclude = {
      association: 'exam',
      attributes: ['title', 'subject', 'level']
    };
    if (subject) {
      examInclude.where = { subject };
    }
    const records = await ExamRecord.findAll({
      include: [examInclude],
      order: [['createdAt', 'DESC']],
      limit: 20,
    });
    res.json(records);
  } catch (err) { next(err); }
});

// ── GET /api/exams/records/:id ───────────────────────────────────────────────
router.get('/records/:id', async (req, res, next) => {
  try {
    const record = await ExamRecord.findByPk(req.params.id, { include: ['exam'] });
    if (!record) return res.status(404).json({ error: 'Record not found' });
    res.json(record);
  } catch (err) { next(err); }
});

router.delete('/records/:id', async (req, res, next) => {
  try {
    const record = await ExamRecord.findByPk(req.params.id);
    if (!record) return res.status(404).json({ error: 'Record not found' });
    await record.destroy();
    res.json({ success: true });
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
    const { answers, timeElapsed } = req.body;
    const exam = await Exam.findByPk(req.params.id);
    if (!exam) return res.status(404).json({ error: 'Exam not found' });

    const sections = exam.sections;
    let earned = 0, total = 0;
    Object.values(sections).forEach((qs) => {
      qs.forEach((q) => {
        total += q.points || 0;
        const ua = (answers[q.id] || '').toString().trim().toLowerCase();
        const ca = (q.answer || '').toString().trim().toLowerCase();
        if (q.type === 'mc' || q.type === 'tf') {
          if (ua === ca) earned += q.points || 0;
        } else {
          if (ua && (ua.includes(ca) || ca.includes(ua))) earned += q.points || 0;
        }
      });
    });

    const score = Math.round((earned / total) * 100);
    await ExamRecord.create({ examId: exam.id, score, total: 100, timeElapsed: timeElapsed || 0, answersJson: JSON.stringify(answers) });
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
      const ok = (q.type === 'mc' || q.type === 'tf') ? ua === ca : (ua.includes(ca) || ca.includes(ua));
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
        { id: 'mc1', type: 'mc', question: 'Which word means "Good morning" in Spanish?', options: ['Buenos días', 'Buenas noches', 'Hola', 'Adiós'], answer: 'A', points: 4 },
        { id: 'mc2', type: 'mc', question: 'What is the plural of "el libro"?', options: ['los libros', 'las libros', 'el libros', 'los libro'], answer: 'A', points: 4 },
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
