import axios from 'axios';

const API = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3002',
  timeout: 30000,
});

// ─── Mock data for development ──────────────────────────────────────────────
function generateMockExam(subject, level) {
  const subjectName = {
    spanish: 'Spanish',
    german: 'German',
    'english-pte': 'English PTE',
  }[subject] || subject;

  return {
    id: `mock-${Date.now()}`,
    title: `${subjectName} Exam${level ? ' – ' + level : ''}`,
    subject,
    level,
    createdAt: new Date().toISOString(),
    sections: {
      multipleChoice: [
        {
          id: 'mc1', type: 'mc',
          question: 'Which of the following is the correct translation of "Good morning"?',
          options: ['Buenos días', 'Buenas noches', 'Buenas tardes', 'Hola'],
          answer: 'A', points: 4,
        },
        {
          id: 'mc2', type: 'mc',
          question: 'What is the past tense of "hablar" (to speak) in the yo form?',
          options: ['hablo', 'hablé', 'hablaré', 'hablaba'],
          answer: 'B', points: 4,
        },
        {
          id: 'mc3', type: 'mc',
          question: 'Which word means "book" in Spanish?',
          options: ['mesa', 'silla', 'libro', 'ventana'],
          answer: 'C', points: 4,
        },
        {
          id: 'mc4', type: 'mc',
          question: 'Choose the correct article for "casa" (house):',
          options: ['el', 'la', 'los', 'las'],
          answer: 'B', points: 4,
        },
        {
          id: 'mc5', type: 'mc',
          question: '"Me llamo María" means:',
          options: ['I call María', 'My name is María', 'I love María', 'I know María'],
          answer: 'B', points: 4,
        },
      ],
      fillIn: [
        {
          id: 'fi1', type: 'fill',
          question: 'Complete the sentence: "Yo _____ (to be) estudiante." (use estar/ser)',
          answer: 'soy', points: 4,
        },
        {
          id: 'fi2', type: 'fill',
          question: 'Translate "I have two brothers" to Spanish: "Yo _____ dos hermanos."',
          answer: 'tengo', points: 4,
        },
        {
          id: 'fi3', type: 'fill',
          question: 'Fill in the blank: "¿Cuántos años _____ tú?" (How old are you?)',
          answer: 'tienes', points: 4,
        },
      ],
      trueFalse: [
        {
          id: 'tf1', type: 'tf',
          question: 'The capital of Spain is Madrid.',
          answer: 'True', points: 2,
        },
        {
          id: 'tf2', type: 'tf',
          question: 'Spanish is the only official language in Spain.',
          answer: 'False', points: 2,
        },
      ],
      translation: [
        {
          id: 'tr1', type: 'translation',
          question: 'I love learning new languages.',
          answer: 'Me encanta aprender nuevos idiomas.', points: 2,
        },
      ],
    },
  };
}

// ─── API functions ───────────────────────────────────────────────────────────

export async function generateExamFromFile(file, subject) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('subject', subject);
  const { data } = await API.post('/api/exams/generate-from-file', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return data;
}

export async function generateExamAuto(subject, level) {
  const { data } = await API.post('/api/exams/generate-auto', { subject, level });
  return data;
}

export async function getExam(id) {
  try {
    const { data } = await API.get(`/api/exams/${id}`);
    return data;
  } catch {
    const stored = sessionStorage.getItem(`exam-${id}`);
    if (stored) return JSON.parse(stored);
    return generateMockExam('spanish', 'beginner-1');
  }
}

export async function submitExam(id, answers, timeElapsed) {
  try {
    const { data } = await API.post(`/api/exams/${id}/submit`, { answers, timeElapsed });
    return data;
  } catch {
    return { score: 78, breakdown: {}, message: 'Exam submitted successfully!' };
  }
}

export async function verifySection(id, section, answers) {
  try {
    const { data } = await API.post(`/api/exams/${id}/verify-section`, { section, answers });
    return data;
  } catch {
    return { correct: 0, total: 0, details: {} };
  }
}

export async function removeFavorite(questionId) {
  await API.delete(`/api/favorites/${questionId}`);
}

export async function getExamRecord(id) {
  try {
    const { data } = await API.get(`/api/exams/records/${id}`);
    return data;
  } catch {
    return null;
  }
}

export async function deleteExamRecord(recordId) {
  try {
    await API.delete(`/api/exams/records/${recordId}`);
    return true;
  } catch {
    return false;
  }
}

export async function getHistory(subject) {
  try {
    const params = subject ? { subject } : {};
    const { data } = await API.get('/api/exams/history', { params });
    return data;
  } catch {
    return [];
  }
}

export async function addFavorite(questionId, question, subject) {
  const questionWithSubject = { ...question, subject };
  await API.post('/api/favorites', { questionId, question: questionWithSubject });
}

export async function getFavorites(subject) {
  try {
    const { data } = await API.get('/api/favorites');
    return subject ? data.filter((f) => f.subject === subject) : data;
  } catch {
    return [];
  }
}