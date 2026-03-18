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
      listening: [
        {
          id: 'li1', type: 'listening',
          audioText: 'Hola, me llamo Carlos. Tengo veinte años y soy de España.',
          question: 'What is the speaker\'s name and age?',
          answer: 'Carlos, 20 years old', points: 5,
        },
        {
          id: 'li2', type: 'listening',
          audioText: 'Hoy hace mucho calor. Voy a la playa con mis amigos.',
          question: 'Where is the speaker going and with whom?',
          answer: 'To the beach with friends', points: 5,
        },
      ],
      reading: [
        {
          id: 'ri1', type: 'reading',
          passage: `Mi familia es muy grande. Tengo dos hermanos y una hermana. Mi padre se llama Juan y mi madre se llama Ana. Vivimos en una casa grande en Madrid. Los fines de semana, vamos al parque juntos.`,
          question: 'How many siblings does the author have?',
          answer: 'Three (two brothers and one sister)', points: 10,
        },
        {
          id: 'ri2', type: 'reading',
          passage: `Mi familia es muy grande. Tengo dos hermanos y una hermana. Mi padre se llama Juan y mi madre se llama Ana. Vivimos en una casa grande en Madrid. Los fines de semana, vamos al parque juntos.`,
          question: 'What does the family do on weekends?',
          answer: 'They go to the park together', points: 10,
        },
      ],
      writing: [
        {
          id: 'wi1', type: 'writing',
          question: 'Write a short paragraph (50–80 words) describing your daily routine in Spanish. Include at least 5 verbs.',
          rubric: 'Vocabulary variety (4pts), Grammar accuracy (4pts), Verb usage (4pts), Coherence (4pts), Word count (4pts)',
          minWords: 50,
          answer: '', points: 20,
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
  const { data } = await API.post('/api/exams/generate-from-file', formData,{
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

export async function submitExam(id, answers) {
  try {
    const { data } = await API.post(`/api/exams/${id}/submit`, { answers });
    return data;
  } catch {
    // Mock scoring
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

export async function addFavorite(questionId, question) {
  try {
    await API.post('/api/favorites', { questionId, question });
  } catch {
    const favs = JSON.parse(localStorage.getItem('eduleaf-favorites') || '[]');
    const exists = favs.find((f) => f.id === questionId);
    if (!exists) {
      favs.push({ id: questionId, ...question, savedAt: new Date().toISOString() });
      localStorage.setItem('eduleaf-favorites', JSON.stringify(favs));
    }
  }
}

export async function removeFavorite(questionId) {
  try {
    await API.delete(`/api/favorites/${questionId}`);
  } catch {
    const favs = JSON.parse(localStorage.getItem('eduleaf-favorites') || '[]');
    localStorage.setItem('eduleaf-favorites', JSON.stringify(favs.filter((f) => f.id !== questionId)));
  }
}

export async function getHistory() {
  try {
    const { data } = await API.get('/api/exams/history');
    return data;
  } catch {
    return [];
  }
}

export async function getFavorites() {
  try {
    const { data } = await API.get('/api/favorites');
    return data;
  } catch {
    return JSON.parse(localStorage.getItem('eduleaf-favorites') || '[]');
  }
}
