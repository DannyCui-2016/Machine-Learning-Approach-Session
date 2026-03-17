"""
Exam Generator Service
Generates structured exam sections from:
  1. Extracted text (file mode)
  2. Subject + difficulty level (auto mode)

Uses rule-based NLP approaches for reliability without requiring API keys.
"""

import re
import random
import uuid
from typing import Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# Auto-generation question banks
# ─────────────────────────────────────────────────────────────────────────────
QUESTION_BANK: Dict[str, Dict[str, Any]] = {
    "spanish": {
        "beginner-1": {
            "multipleChoice": [
                {"q": 'What does "Hola" mean?', "opts": ["Hello","Goodbye","Thank you","Please"], "ans": "A"},
                {"q": 'Which article goes with "libro" (book)?', "opts": ["la","los","el","las"], "ans": "C"},
                {"q": '"Buenos días" means:', "opts": ["Good evening","Good afternoon","Good morning","Goodnight"], "ans": "C"},
                {"q": 'How do you say "Thank you" in Spanish?', "opts": ["Por favor","De nada","Gracias","Perdón"], "ans": "C"},
                {"q": '"Me llamo" means:', "opts": ["I want","My name is","I have","I am"], "ans": "B"},
            ],
            "fillIn": [
                {"q": 'Complete: "Yo _____ estudiante." (to be)', "ans": "soy"},
                {"q": '"¿Cómo _____ tú?" (to be called)', "ans": "te llamas"},
                {"q": 'Translate: "I have a dog." → "Yo _____ un perro."', "ans": "tengo"},
            ],
            "listening": [
                {"audio": "Hola, me llamo Pedro. Tengo diez años.", "q": "What is the speaker's name and age?", "ans": "Pedro, 10 years old"},
                {"audio": "Yo vivo en una casa blanca.", "q": "What colour is the speaker's house?", "ans": "White"},
            ],
            "reading": [
                {"passage": "Ana es una chica de España. Ella tiene doce años y tiene dos gatos. Su color favorito es el verde.", "q": "How many cats does Ana have?", "ans": "Two"},
                {"passage": "Ana es una chica de España. Ella tiene doce años y tiene dos gatos. Su color favorito es el verde.", "q": "What is Ana's favourite colour?", "ans": "Green"},
            ],
            "writing": [
                {"q": "Write 50–80 words about yourself in Spanish. Include name, age, and hobbies.", "rubric": "Vocabulary (5), Grammar (5), Verbs (5), Coherence (5)", "minWords": 50},
            ],
        },
        "intermediate-1": {
            "multipleChoice": [
                {"q": 'What is the preterite of "hablar" for "yo"?', "opts": ["hablé","hablaba","hablaré","hablo"], "ans": "A"},
                {"q": 'Choose the correct form: "Nosotros _____ en Madrid." (vivir)', "opts": ["vivemos","vivimos","viven","vive"], "ans": "B"},
                {"q": '"Se lo dije" uses which pronoun to replace "a él"?', "opts": ["la","le","lo","se"], "ans": "D"},
                {"q": 'Which tense: "Cuando era niño, jugaba al fútbol."', "opts": ["Preterite","Imperfect","Future","Present"], "ans": "B"},
                {"q": '"Aunque" is used to express:', "opts": ["Because","In order to","Although","Therefore"], "ans": "C"},
            ],
            "fillIn": [
                {"q": 'Complete with subjunctive: "Espero que tú _____ bien." (estar)', "ans": "estés"},
                {"q": "Translate: 'I would have gone' → Yo _____ ido.", "ans": "habría"},
                {"q": '"Hace tres años que vivo aquí." translates to: "I _____ here for three years."', "ans": "have lived"},
            ],
            "listening": [
                {"audio": "Ayer fui al mercado y compré frutas y verduras. Había mucha gente y el ambiente era animado.", "q": "Where did the speaker go, and what did they buy?", "ans": "To the market, bought fruits and vegetables"},
                {"audio": "Si hubiera sabido que llovería, habría traído un paraguas.", "q": "What would the speaker have done differently?", "ans": "Brought an umbrella"},
            ],
            "reading": [
                {"passage": "El cambio climático es uno de los mayores desafíos del siglo XXI. Los científicos advierten que si no reducimos las emisiones de carbono, las consecuencias serán devastadoras.", "q": "What do scientists warn about?", "ans": "If carbon emissions are not reduced, the consequences will be devastating"},
            ],
            "writing": [
                {"q": "Write a 100–150 word essay in Spanish arguing for or against social media use by teenagers.", "rubric": "Argument (5), Evidence (5), Grammar (5), Vocabulary (5)", "minWords": 100},
            ],
        },
    },
    "german": {
        "beginner-1": {
            "multipleChoice": [
                {"q": 'What does "Guten Morgen" mean?', "opts": ["Good night","Good morning","Good evening","Goodbye"], "ans": "B"},
                {"q": 'The German word for "dog" is:', "opts": ["Katze","Vogel","Hund","Fisch"], "ans": "C"},
                {"q": '"Ich heiße" means:', "opts": ["I want","My name is","I have","I like"], "ans": "B"},
                {"q": 'Which article goes with "Buch" (book)?', "opts": ["der","die","das","den"], "ans": "C"},
                {"q": '"Wie geht es Ihnen?" means:', "opts": ["What is your name?","How old are you?","How are you?","Where are you from?"], "ans": "C"},
            ],
            "fillIn": [
                {"q": 'Complete: "Ich _____ aus Deutschland." (to come from)', "ans": "komme"},
                {"q": '"Das _____ ein Buch." (to be)', "ans": "ist"},
                {"q": 'Translate: "I am 14 years old." → "Ich _____ 14 Jahre alt."', "ans": "bin"},
            ],
            "listening": [
                {"audio": "Hallo! Ich heiße Mia. Ich komme aus Berlin und bin zwölf Jahre alt.", "q": "What is the speaker's name, city, and age?", "ans": "Mia, Berlin, 12"},
            ],
            "reading": [
                {"passage": "Max wohnt in München. Er hat einen Bruder und eine Schwester. Sein Lieblingstier ist der Hund.", "q": "How many siblings does Max have?", "ans": "Two (a brother and a sister)"},
            ],
            "writing": [
                {"q": "Write 50–80 words about your family in German.", "rubric": "Vocabulary (5), Grammar (5), Verbs (5), Coherence (5)", "minWords": 50},
            ],
        },
    },
    "english-pte": {
        "beginner-1": {
            "multipleChoice": [
                {"q": "Choose the correct article: '_____ umbrella.'", "opts": ["A","An","The","Some"], "ans": "B"},
                {"q": "Which sentence is grammatically correct?", "opts": ["She go to school.","She goes to school.","She going to school.","She goed to school."], "ans": "B"},
                {"q": '"Their" is used to show:', "opts": ["Location","Possession","Action","Time"], "ans": "B"},
                {"q": "The synonym of 'happy' is:", "opts": ["Sad","Angry","Joyful","Tired"], "ans": "C"},
                {"q": "Choose the correct sentence:", "opts": ["I have been here since 3 hours.","I have been here for 3 hours.","I was here since 3 hours.","I have here for 3 hours."], "ans": "B"},
            ],
            "fillIn": [
                {"q": "She _____ to school every day. (go – present simple)", "ans": "goes"},
                {"q": "By next year, I _____ been here for a decade. (will – future perfect)", "ans": "will have"},
            ],
            "listening": [
                {"audio": "The Great Barrier Reef is the world's largest coral reef system, found off the coast of Australia.", "q": "Where is the Great Barrier Reef located?", "ans": "Off the coast of Australia"},
            ],
            "reading": [
                {"passage": "Photosynthesis is the process by which green plants convert sunlight into food. They absorb carbon dioxide and water, releasing oxygen as a by-product.", "q": "What do plants release during photosynthesis?", "ans": "Oxygen"},
            ],
            "writing": [
                {"q": "Write a 100–150 word response: 'Do you agree that technology has improved our lives?' Give reasons.", "rubric": "Content (5), Language (5), Grammar (5), Structure (5)", "minWords": 100},
            ],
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Points allocation by question type
# ─────────────────────────────────────────────────────────────────────────────
POINTS = {
    "multipleChoice": 4,
    "fillIn": 4,
    "listening": 5,
    "reading": 10,
    "writing": 20,
}


def _make_id() -> str:
    return str(uuid.uuid4())[:8]


def _format_section(raw_questions: list, type_key: str) -> list:
    """Convert raw bank entries to standard question objects."""
    formatted = []
    for q in raw_questions:
        if type_key == "multipleChoice":
            formatted.append({
                "id":       _make_id(),
                "type":     "mc",
                "question": q["q"],
                "options":  q["opts"],
                "answer":   q["ans"],
                "points":   POINTS[type_key],
            })
        elif type_key == "fillIn":
            formatted.append({
                "id":       _make_id(),
                "type":     "fill",
                "question": q["q"],
                "answer":   q["ans"],
                "points":   POINTS[type_key],
            })
        elif type_key == "listening":
            formatted.append({
                "id":        _make_id(),
                "type":      "listening",
                "audioText": q["audio"],
                "question":  q["q"],
                "answer":    q["ans"],
                "points":    POINTS[type_key],
            })
        elif type_key == "reading":
            formatted.append({
                "id":       _make_id(),
                "type":     "reading",
                "passage":  q["passage"],
                "question": q["q"],
                "answer":   q["ans"],
                "points":   POINTS[type_key],
            })
        elif type_key == "writing":
            formatted.append({
                "id":       _make_id(),
                "type":     "writing",
                "question": q["q"],
                "rubric":   q.get("rubric", ""),
                "minWords": q.get("minWords", 50),
                "answer":   "",
                "points":   POINTS[type_key],
            })
    return formatted


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def generate_auto(subject: str, level: str) -> dict:
    """Generate a full exam for the given subject and level from the question bank."""
    bank = QUESTION_BANK.get(subject, {})
    level_data = bank.get(level, bank.get("beginner-1", {}))

    sections = {}
    for section_key in ["multipleChoice", "fillIn", "listening", "reading", "writing"]:
        raw = level_data.get(section_key, [])
        if raw:
            sections[section_key] = _format_section(raw, section_key)

    subject_names = {"spanish": "Spanish", "german": "German", "english-pte": "English PTE"}
    return {
        "title":    f"{subject_names.get(subject, subject)} – {level.title()}",
        "subject":  subject,
        "level":    level,
        "sections": sections,
    }


def generate_from_text(text: str, subject: str) -> dict:
    """
    Generate exam questions from extracted document text.
    Uses rule-based NLP to extract sentences and form questions.
    This is a deterministic baseline; can be upgraded to LLM calls.
    """
    sentences = _extract_sentences(text)
    random.shuffle(sentences)

    mc_questions      = _build_mc_from_sentences(sentences[:10], subject)
    fill_questions    = _build_fill_from_sentences(sentences[10:16])
    reading_questions = _build_reading(sentences[16:30])
    writing_questions = _build_writing(subject)

    subject_names = {"spanish": "Spanish", "german": "German", "english-pte": "English PTE"}
    return {
        "title":   f"{subject_names.get(subject, subject)} – Document Exam",
        "subject": subject,
        "source":  "file",
        "sections": {
            "multipleChoice": mc_questions,
            "fillIn":         fill_questions,
            "reading":        reading_questions,
            "writing":        writing_questions,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# NLP helpers
# ─────────────────────────────────────────────────────────────────────────────
def _extract_sentences(text: str) -> list:
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if 20 < len(s.strip()) < 300]


def _build_mc_from_sentences(sentences: list, subject: str) -> list:
    """
    Create multiple-choice questions by asking what a sentence implies.
    In production, replace with LLM call for higher quality.
    """
    qs = []
    for i, sent in enumerate(sentences[:5]):
        # Create a simple "what does this passage discuss" stub
        snippet = sent[:80] + ("…" if len(sent) > 80 else "")
        opts_pool = [
            "Grammar rules",
            "Vocabulary definitions",
            "Cultural context",
            "Pronunciation patterns",
            "Historical events",
            "Scientific concepts",
        ]
        random.shuffle(opts_pool)
        qs.append({
            "id":       _make_id(),
            "type":     "mc",
            "question": f'According to the text, the phrase "{snippet}" relates to which category?',
            "options":  opts_pool[:4],
            "answer":   "A",   # First option is considered correct in this stub
            "points":   4,
        })
    return qs


def _build_fill_from_sentences(sentences: list) -> list:
    """Create fill-in-the-blank by masking the last significant word of a sentence."""
    qs = []
    for sent in sentences[:3]:
        words = sent.split()
        if len(words) < 5:
            continue
        # Mask the last meaningful word (not punctuation)
        target_idx = len(words) - 1
        while target_idx > 0 and not words[target_idx].isalpha():
            target_idx -= 1
        if target_idx > 0:
            answer = words[target_idx]
            masked = words.copy()
            masked[target_idx] = "_____"
            qs.append({
                "id":       _make_id(),
                "type":     "fill",
                "question": " ".join(masked),
                "answer":   answer,
                "points":   4,
            })
    return qs


def _build_reading(sentences: list) -> list:
    if not sentences:
        return []
    passage = " ".join(sentences[:8])
    return [
        {
            "id":       _make_id(),
            "type":     "reading",
            "passage":  passage,
            "question": "In your own words, summarise the main idea of the passage.",
            "answer":   "(open-ended – evaluator assessment)",
            "points":   10,
        },
        {
            "id":       _make_id(),
            "type":     "reading",
            "passage":  passage,
            "question": "Identify one example or piece of evidence the author provides.",
            "answer":   "(open-ended – evaluator assessment)",
            "points":   10,
        },
    ]


def _build_writing(subject: str) -> list:
    prompts = {
        "spanish":      "Write 80–100 words in Spanish about the main theme of the document you uploaded.",
        "german":       "Write 80–100 words in German summarising what you learned from the document.",
        "english-pte":  "Write a 100–150 word response to the ideas presented in the document.",
    }
    return [{
        "id":       _make_id(),
        "type":     "writing",
        "question": prompts.get(subject, "Write a short essay based on the document content."),
        "rubric":   "Content relevance (5), Grammar (5), Vocabulary (5), Structure (5)",
        "minWords": 80,
        "answer":   "",
        "points":   20,
    }]
