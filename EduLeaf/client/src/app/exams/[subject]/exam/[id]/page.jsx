'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import { useLanguage } from '../../../../../contexts/LanguageContext';
import { getExam, submitExam, verifySection, addFavorite, removeFavorite } from '../../../../../services/examService';
import styles from './page.module.css';

const SECTIONS = [
  { key: 'multipleChoice', icon: '☑️', labelKey: 'exam_page.multiple_choice' },
  { key: 'fillIn',         icon: '✏️', labelKey: 'exam_page.fill_in'         },
  { key: 'listening',      icon: '🎧', labelKey: 'exam_page.listening'       },
  { key: 'reading',        icon: '📖', labelKey: 'exam_page.reading'         },
  { key: 'writing',        icon: '📝', labelKey: 'exam_page.writing'         },
];

export default function ExamPage() {
  const { subject, id } = useParams();
  const { t } = useLanguage();

  const [exam, setExam]           = useState(null);
  const [loading, setLoading]     = useState(true);
  const [answers, setAnswers]     = useState({});
  const [favorites, setFavorites] = useState(new Set());
  const [verified, setVerified]   = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [result, setResult]       = useState(null);
  const [activeSection, setActiveSection] = useState('multipleChoice');
  const [sectionVerifyResult, setSectionVerifyResult] = useState({});

  const sectionRefs = useRef({});
  const navRef = useRef(null);

  // ── Load exam ──────────────────────────────────────────────────────────────
  useEffect(() => {
    (async () => {
      try {
        const data = await getExam(id);
        setExam(data);
      } finally {
        setLoading(false);
      }
    })();

    // Restore favorites from localStorage
    const storedFavs = JSON.parse(localStorage.getItem('eduleaf-favorites') || '[]');
    setFavorites(new Set(storedFavs.map((f) => f.id)));
  }, [id]);

  // ── Scroll spy ─────────────────────────────────────────────────────────────
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.dataset.section);
          }
        });
      },
      { rootMargin: '-20% 0px -70% 0px', threshold: 0 }
    );

    SECTIONS.forEach(({ key }) => {
      const el = sectionRefs.current[key];
      if (el) observer.observe(el);
    });
    return () => observer.disconnect();
  }, [exam]);

  const scrollTo = (key) => {
    sectionRefs.current[key]?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  // ── Answers ────────────────────────────────────────────────────────────────
  const setAnswer = useCallback((qId, value) => {
    setAnswers((prev) => ({ ...prev, [qId]: value }));
  }, []);

  // ── Favorites ──────────────────────────────────────────────────────────────
  const toggleFavorite = useCallback(async (q) => {
    const isFav = favorites.has(q.id);
    const next = new Set(favorites);
    if (isFav) {
      next.delete(q.id);
      await removeFavorite(q.id);
    } else {
      next.add(q.id);
      await addFavorite(q.id, q);
    }
    setFavorites(next);
  }, [favorites]);

  // ── Verify section ─────────────────────────────────────────────────────────
  const handleVerifySection = useCallback(async (sectionKey) => {
    if (!exam) return;
    const sectionQs = exam.sections[sectionKey] || [];
    let correct = 0;
    const details = {};
    sectionQs.forEach((q) => {
      const userAns = (answers[q.id] || '').toString().trim().toLowerCase();
      const correctAns = (q.answer || '').toString().trim().toLowerCase();
      const isOk = q.type === 'mc'
        ? userAns === correctAns
        : userAns.includes(correctAns) || correctAns.includes(userAns);
      details[q.id] = isOk;
      if (isOk) correct++;
    });
    setSectionVerifyResult((prev) => ({ ...prev, [sectionKey]: details }));
    setVerified((prev) => ({ ...prev, [sectionKey]: { correct, total: sectionQs.length } }));
  }, [exam, answers]);

  // ── Submit exam ────────────────────────────────────────────────────────────
  const handleSubmit = useCallback(async () => {
    if (!exam) return;
    const res = await submitExam(exam.id, answers);
    // Calculate client-side score
    let earned = 0;
    let total = 0;
    Object.values(exam.sections).forEach((qs) => {
      qs.forEach((q) => {
        total += q.points || 0;
        const userAns = (answers[q.id] || '').toString().trim().toLowerCase();
        const correctAns = (q.answer || '').toString().trim().toLowerCase();
        if (q.type === 'writing') {
          // Writing: award half points if answered
          if (userAns.split(' ').length >= (q.minWords || 0) * 0.5) earned += (q.points || 0) * 0.6;
        } else if (q.type === 'mc') {
          if (userAns === correctAns) earned += q.points || 0;
        } else {
          if (userAns && (userAns.includes(correctAns) || correctAns.includes(userAns))) earned += q.points || 0;
        }
      });
    });
    const score = Math.round((earned / total) * 100);
    setResult({ score, total: 100 });
    setSubmitted(true);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [exam, answers]);

  // ── Render helpers ─────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className={styles.loadingWrap}>
        <span className="spinner" />
        <p>{t('common.loading')}</p>
      </div>
    );
  }

  if (!exam) {
    return (
      <div className={styles.loadingWrap}>
        <p>{t('common.error')}</p>
        <Link href={`/exams/${subject}`} className="btn btn-primary">← {t('common.back')}</Link>
      </div>
    );
  }

  return (
    <div className={styles.page}>
      {/* ── Score Banner ── */}
      {submitted && result && (
        <div className={styles.scoreBanner}>
          <div className={styles.scoreBannerInner}>
            <span className={styles.scoreTrophy}>🏆</span>
            <div>
              <h2 className={styles.scoreTitle}>{t('exam_page.score_result')}</h2>
              <div className={styles.scoreNum}>
                <span className={styles.scoreBig}>{result.score}</span>
                <span className={styles.scoreOf}>/ 100</span>
              </div>
              <p className={styles.scoreMsg}>
                {result.score >= 90 ? '🌟 Excellent!' : result.score >= 70 ? '👍 Well done!' : '💪 Keep practising!'}
              </p>
            </div>
            <div className={styles.scoreActions}>
              <Link href={`/exams/${subject}`} className="btn btn-secondary">{t('common.back')}</Link>
              <button onClick={() => { setSubmitted(false); setResult(null); setAnswers({}); setSectionVerifyResult({}); setVerified({}); }} className="btn btn-primary">
                {t('exam.take_exam_btn')}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Exam Header ── */}
      <div className={styles.examHeader}>
        <div className="container">
          <div className={styles.examHeaderRow}>
            <Link href={`/exams/${subject}`} className={styles.back}>← {t('common.back')}</Link>
            <div>
              <h1 className={styles.examTitle}>{exam.title}</h1>
              <p className={styles.examMeta}>
                {exam.subject} · {exam.level || 'Custom'} ·{' '}
                {Object.values(exam.sections).reduce((s, arr) => s + arr.length, 0)} questions · 100 pts
              </p>
            </div>
            {!submitted && (
              <button className={`btn btn-primary ${styles.submitBtn}`} onClick={handleSubmit}>
                🎯 {t('exam_page.submit_exam')}
              </button>
            )}
          </div>
        </div>
      </div>

      <div className="container">
        <div className={styles.examLayout}>
          {/* ── Sticky Side Nav ── */}
          <nav className={styles.sideNav} ref={navRef}>
            <p className={styles.sideNavTitle}>Sections</p>
            {SECTIONS.map(({ key, icon, labelKey }) => {
              const qs = exam.sections[key];
              if (!qs || qs.length === 0) return null;
              return (
                <button
                  key={key}
                  className={`${styles.navItem} ${activeSection === key ? styles.navItemActive : ''}`}
                  onClick={() => scrollTo(key)}
                >
                  <span className={styles.navIcon}>{icon}</span>
                  <span className={styles.navLabel}>{t(labelKey)}</span>
                  {verified[key] && (
                    <span className={styles.navBadge}>
                      {verified[key].correct}/{verified[key].total}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>

          {/* ── Exam Content ── */}
          <div className={styles.examContent}>
            {SECTIONS.map(({ key, icon, labelKey }) => {
              const qs = exam.sections[key];
              if (!qs || qs.length === 0) return null;
              const sectionVR = sectionVerifyResult[key] || {};

              return (
                <section
                  key={key}
                  className={styles.section}
                  ref={(el) => { sectionRefs.current[key] = el; }}
                  data-section={key}
                >
                  <div className={styles.sectionHeader}>
                    <div className={styles.sectionTitleRow}>
                      <span className={styles.sectionIcon}>{icon}</span>
                      <h2 className={styles.sectionTitle}>{t(labelKey)}</h2>
                      <span className={styles.sectionCount}>{qs.length} questions</span>
                    </div>
                    {!submitted && (
                      <button
                        className={`btn btn-secondary btn-sm ${styles.verifyBtn}`}
                        onClick={() => handleVerifySection(key)}
                      >
                        ✓ {t('exam_page.verify_section')}
                      </button>
                    )}
                  </div>

                  {verified[key] && (
                    <div className={styles.sectionResult}>
                      ✅ {verified[key].correct} / {verified[key].total} correct in this section
                    </div>
                  )}

                  <div className={styles.questions}>
                    {qs.map((q, qIdx) => (
                      <QuestionCard
                        key={q.id}
                        q={q}
                        qIdx={qIdx}
                        sectionKey={key}
                        answers={answers}
                        setAnswer={setAnswer}
                        favorites={favorites}
                        toggleFavorite={toggleFavorite}
                        verifyResult={sectionVR[q.id]}
                        submitted={submitted}
                        t={t}
                      />
                    ))}
                  </div>
                </section>
              );
            })}

            {/* Final Submit */}
            {!submitted && (
              <div className={styles.finalSubmit}>
                <p className={styles.finalHint}>
                  🎉 Ready? Review your answers above, then submit for your final score.
                </p>
                <button className="btn btn-primary btn-lg" onClick={handleSubmit}>
                  🎯 {t('exam_page.submit_exam')}
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Question Card component ────────────────────────────────────────────────────
function QuestionCard({ q, qIdx, sectionKey, answers, setAnswer, favorites, toggleFavorite, verifyResult, submitted, t }) {
  const isFav = favorites.has(q.id);
  const userAns = answers[q.id] || '';
  const hasVerify = verifyResult !== undefined;

  return (
    <div
      className={`${styles.questionCard} ${
        hasVerify ? (verifyResult ? styles.questionCorrect : styles.questionWrong) : ''
      }`}
    >
      <div className={styles.questionHeader}>
        <span className={styles.questionNum}>Q{qIdx + 1}</span>
        <span className={styles.questionPoints}>{q.points} pts</span>
        <button
          className={`${styles.favBtn} ${isFav ? styles.favBtnActive : ''}`}
          onClick={() => toggleFavorite(q)}
          title={isFav ? t('exam_page.unfavorite') : t('exam_page.favorite')}
        >
          {isFav ? '⭐' : '☆'}
        </button>
      </div>

      {/* Reading passage */}
      {q.passage && (
        <div className={styles.passage}>{q.passage}</div>
      )}

      {/* Listening audio */}
      {q.audioText && (
        <ListeningPlayer audioText={q.audioText} t={t} />
      )}

      <p className={styles.questionText}>{q.question}</p>

      {/* Multiple Choice */}
      {q.type === 'mc' && (
        <div className={styles.optionsGrid}>
          {q.options.map((opt, i) => {
            const letter = String.fromCharCode(65 + i);
            const isSelected = userAns === letter;
            const isCorrect = letter === q.answer;
            return (
              <button
                key={letter}
                disabled={submitted}
                className={`${styles.optionBtn} ${isSelected ? styles.optionSelected : ''} ${
                  hasVerify && isCorrect ? styles.optionCorrect : ''
                } ${hasVerify && isSelected && !isCorrect ? styles.optionWrong : ''}`}
                onClick={() => setAnswer(q.id, letter)}
              >
                <span className={styles.optionLetter}>{letter}</span>
                <span>{opt}</span>
              </button>
            );
          })}
        </div>
      )}

      {/* Fill in the blank */}
      {q.type === 'fill' && (
        <div className={styles.fillWrap}>
          <input
            className={`form-input ${styles.fillInput} ${
              hasVerify ? (verifyResult ? styles.inputCorrect : styles.inputWrong) : ''
            }`}
            type="text"
            placeholder={t('exam_page.your_answer')}
            value={userAns}
            onChange={(e) => setAnswer(q.id, e.target.value)}
            disabled={submitted}
          />
        </div>
      )}

      {/* Listening / Reading – text answer */}
      {(q.type === 'listening' || q.type === 'reading') && (
        <div className={styles.fillWrap}>
          <textarea
            className={`form-textarea ${styles.textAnswer} ${
              hasVerify ? (verifyResult ? styles.inputCorrect : styles.inputWrong) : ''
            }`}
            placeholder={t('exam_page.your_answer')}
            value={userAns}
            onChange={(e) => setAnswer(q.id, e.target.value)}
            disabled={submitted}
            rows={3}
          />
        </div>
      )}

      {/* Writing */}
      {q.type === 'writing' && (
        <div>
          {q.rubric && <div className={styles.rubric}>📋 Rubric: {q.rubric}</div>}
          <textarea
            className={`form-textarea ${styles.writingAnswer}`}
            placeholder={`Write at least ${q.minWords || 50} words…`}
            value={userAns}
            onChange={(e) => setAnswer(q.id, e.target.value)}
            disabled={submitted}
            rows={8}
          />
          <div className={styles.wordCount}>
            Words: {(userAns || '').split(/\s+/).filter(Boolean).length} / {q.minWords || 50} min
          </div>
        </div>
      )}

      {/* Show correct answer after verify */}
      {hasVerify && !verifyResult && q.answer && q.type !== 'writing' && (
        <div className={styles.modelAnswer}>
          💡 {t('exam_page.model_answer')}: <strong>{q.answer}</strong>
        </div>
      )}

      {hasVerify && (
        <div className={`${styles.verifyBadge} ${verifyResult ? styles.verifyCorrect : styles.verifyWrong}`}>
          {verifyResult ? `✅ ${t('exam_page.answer_correct')}` : `❌ ${t('exam_page.answer_wrong')}`}
        </div>
      )}
    </div>
  );
}

// ── Listening player (uses browser TTS) ────────────────────────────────────────
function ListeningPlayer({ audioText, t }) {
  const [playing, setPlaying] = useState(false);

  const play = () => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(audioText);
    utt.rate = 0.9;
    utt.onstart = () => setPlaying(true);
    utt.onend = () => setPlaying(false);
    window.speechSynthesis.speak(utt);
  };

  const stop = () => {
    window.speechSynthesis?.cancel();
    setPlaying(false);
  };

  return (
    <div className={styles.audioPlayer}>
      <span className={styles.audioIcon}>🎧</span>
      <div className={styles.audioWave}>
        {[1,2,3,4,5].map((i) => (
          <span key={i} className={`${styles.audioBar} ${playing ? styles.audioBarPlaying : ''}`} style={{ animationDelay: `${i * 0.1}s` }} />
        ))}
      </div>
      <button
        className={`btn btn-secondary btn-sm ${styles.playBtn}`}
        onClick={playing ? stop : play}
      >
        {playing ? '⏹ Stop' : `▶ ${t('exam_page.listen_btn')}`}
      </button>
    </div>
  );
}
