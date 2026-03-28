'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import Link from 'next/link';
import { useParams, useSearchParams } from 'next/navigation';
import { useLanguage } from '../../../../../contexts/LanguageContext';
import { getExam, getExamRecord, submitExam, verifySection, addFavorite, removeFavorite, getFavorites } from '../../../../../services/examService';
import styles from './page.module.css';

const SECTIONS = [
  { key: 'multipleChoice', icon: '☑️', labelKey: 'exam_page.multiple_choice' },
  { key: 'fillIn', icon: '✏️', labelKey: 'exam_page.fill_in' },
  { key: 'trueFalse', icon: '⚖️', labelKey: 'exam_page.true_false' },
  { key: 'translation', icon: '🌐', labelKey: 'exam_page.translation' },
];

export default function ExamPage() {
  const { subject, id } = useParams();
  const searchParams = useSearchParams();
  const recordId = searchParams.get('recordId');
  const { t } = useLanguage();

  const [exam, setExam] = useState(null);
  const [loading, setLoading] = useState(true);
  const [answers, setAnswers] = useState({});
  const [favorites, setFavorites] = useState(new Set());
  const [verified, setVerified] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [result, setResult] = useState(null);
  const [activeSection, setActiveSection] = useState('multipleChoice');
  const [sectionVerifyResult, setSectionVerifyResult] = useState({});

  const sectionRefs = useRef({});
  const navRef = useRef(null);

  const [timeElapsed, setTimeElapsed] = useState(0);
  const timerRef = useRef(null);

  useEffect(() => {
    timerRef.current = setInterval(() => {
      setTimeElapsed((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(timerRef.current);
  }, []);

  const formatTime = (seconds) => {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };
  // ── Load exam ──────────────────────────────────────────────────────────────
  useEffect(() => {
    (async () => {
      try {
        const data = await getExam(id);
        setExam(data);

        const storedFavs = await getFavorites(subject);
        setFavorites(new Set(storedFavs.map((f) => f.id)));

        if (recordId) {
          const record = await getExamRecord(recordId);
          if (record && record.answersJson) {
            const ans = JSON.parse(record.answersJson);
            setAnswers(ans);
            reviewAnswers(data, ans);
          }
        }
      } finally {
        setLoading(false);
      }
    })();
  }, [id]);

  // ── Scroll spy ─────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!exam) return;

    const handleScroll = () => {
      const scrollY = window.scrollY + 200;
      let current = 'multipleChoice';
      SECTIONS.forEach(({ key }) => {
        const el = sectionRefs.current[key];
        if (el && el.offsetTop <= scrollY) {
          current = key;
        }
      });
      setActiveSection(current);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener('scroll', handleScroll);
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
      await addFavorite(q.id, q, subject);
    }
    setFavorites(next);
  }, [favorites, subject]);

  const reviewAnswers = useCallback((examData, ans) => {
    let earned = 0;
    let total = 0;
    const newSectionVerifyResult = {};
    const newVerified = {};

    const normalize = (str) =>
      (str || '')
        .toLowerCase()
        .replace(/['’`´]/g, "'")
        .replace(/[.,!?¡¿"“”]/g, "")
        .replace(/\s+/g, " ")
        .trim();

    Object.entries(examData.sections).forEach(([sectionKey, qs]) => {
      let correct = 0;
      const details = {};
      qs.forEach((q) => {
        total += q.points || 0;
        const userAns = (ans[q.id] || '').toString().trim().toLowerCase();
        const correctAns = (q.answer || '').toString().trim().toLowerCase();
        
        const normUser = normalize(ans[q.id]);
        const normCorrect = normalize(q.answer);

        const isOk = (q.type === 'mc' || q.type === 'tf')
          ? userAns === correctAns
          : !!normUser && (normUser.includes(normCorrect) || normCorrect.includes(normUser));

        details[q.id] = isOk;
        if (isOk) {
          correct++;
          earned += q.points || 0;
        }
      });
      newSectionVerifyResult[sectionKey] = details;
      newVerified[sectionKey] = { correct, total: qs.length };
    });

    setSectionVerifyResult(newSectionVerifyResult);
    setVerified(newVerified);
    const score = Math.round((earned / total) * 100);
    setResult({ score, total: 100 });
    setSubmitted(true);
  }, []);

  // ── Verify section ─────────────────────────────────────────────────────────
  const handleVerifySection = useCallback(async (sectionKey) => {
    if (!exam) return;
    try {
      const result = await verifySection(exam.id, sectionKey, answers);
      setSectionVerifyResult((prev) => ({ ...prev, [sectionKey]: result.details }));
      setVerified((prev) => ({ ...prev, [sectionKey]: { correct: result.correct, total: result.total } }));
    } catch (err) {
      console.error(err);
    }
  }, [exam, answers]);

  // ── Submit exam ────────────────────────────────────────────────────────────
  const handleSubmit = useCallback(async () => {
    if (!exam) return;
    clearInterval(timerRef.current);
    await submitExam(exam.id, answers, timeElapsed);
    reviewAnswers(exam, answers);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [exam, answers, reviewAnswers, timeElapsed]);

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
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginLeft: 'auto' }}>
                <div style={{
                  fontSize: '0.9rem',
                  color: 'var(--color-primary-deep)',
                  fontWeight: '600',
                  fontVariantNumeric: 'tabular-nums',
                  background: 'var(--color-primary-pale)',
                  padding: '6px 14px',
                  borderRadius: '999px',
                  border: '1px solid var(--color-primary-lighter)',
                }}>
                  ⏱ {formatTime(timeElapsed)}
                </div>
                {process.env.NODE_ENV === 'development' && (
                  <button
                    className={`btn btn-secondary ${styles.submitBtn}`}
                    onClick={() => {
                      const mockAns = {};
                      Object.values(exam.sections).forEach(qs => {
                        qs.forEach(q => {
                          mockAns[q.id] = (q.type === 'mc') ? 'A' : (q.type === 'tf') ? 'true' : (q.answer || 'mock text');
                        });
                      });
                      setAnswers(mockAns);
                    }}
                    title="Dev Only: Auto-fill answers"
                  >
                    ⚡ Auto-Fill
                  </button>
                )}
                <button className={`btn btn-primary ${styles.submitBtn}`} onClick={handleSubmit}>
                  🎯 {t('exam_page.submit_exam')}
                </button>
              </div>
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
                <div style={{ display: 'flex', gap: '10px', justifyContent: 'center' }}>
                  {process.env.NODE_ENV === 'development' && (
                    <button
                      className="btn btn-secondary btn-lg"
                      onClick={() => {
                        const mockAns = {};
                        Object.values(exam.sections).forEach(qs => {
                          qs.forEach(q => {
                            mockAns[q.id] = (q.type === 'mc') ? 'A' : (q.type === 'tf') ? 'true' : (q.answer || 'mock text');
                          });
                        });
                        setAnswers(mockAns);
                        window.scrollTo({ top: 0, behavior: 'smooth' });
                      }}
                    >
                      ⚡ Auto-Fill
                    </button>
                  )}
                  <button className="btn btn-primary btn-lg" onClick={handleSubmit}>
                    🎯 {t('exam_page.submit_exam')}
                  </button>
                </div>
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
      className={`${styles.questionCard} ${hasVerify ? (verifyResult ? styles.questionCorrect : styles.questionWrong) : ''
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
        <div className={styles.passage}>{q.passage?.split('\n')[0]}</div>
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
                className={`${styles.optionBtn} ${isSelected ? styles.optionSelected : ''} ${hasVerify && isCorrect ? styles.optionCorrect : ''
                  } ${hasVerify && isSelected && !isCorrect ? styles.optionWrong : ''}`}
                onClick={() => setAnswer(q.id, letter)}
              >
                <span className={styles.optionLetter}>{letter}</span>
                <span>{opt.replace(/^[A-D][.)]\s*/i, '')}</span>
              </button>
            );
          })}
        </div>
      )}

      {/* Fill in the blank */}
      {q.type === 'fill' && (
        <div className={styles.fillWrap}>
          <input
            className={`form-input ${styles.fillInput} ${hasVerify ? (verifyResult ? styles.inputCorrect : styles.inputWrong) : ''
              }`}
            type="text"
            placeholder={t('exam_page.your_answer')}
            value={userAns}
            onChange={(e) => setAnswer(q.id, e.target.value)}
            disabled={submitted}
          />
        </div>
      )}

      {/* True / False */}
      {q.type === 'tf' && (
        <div className={styles.optionsGrid}>
          {['True', 'False'].map((opt) => {
            const isSelected = userAns === opt.toLowerCase();
            const isCorrect = opt.toLowerCase() === (q.answer || '').toLowerCase();
            return (
              <button
                key={opt}
                disabled={submitted}
                className={`${styles.optionBtn} ${isSelected ? styles.optionSelected : ''} ${hasVerify && isCorrect ? styles.optionCorrect : ''
                  } ${hasVerify && isSelected && !isCorrect ? styles.optionWrong : ''}`}
                onClick={() => setAnswer(q.id, opt.toLowerCase())}
              >
                <span>{opt}</span>
              </button>
            );
          })}
        </div>
      )}

      {/* Translation */}
      {q.type === 'translation' && (
        <div className={styles.fillWrap}>
          <textarea
            className={`form-textarea ${styles.textAnswer} ${hasVerify ? (verifyResult ? styles.inputCorrect : styles.inputWrong) : ''
              }`}
            placeholder={t('exam_page.your_answer')}
            value={userAns}
            onChange={(e) => setAnswer(q.id, e.target.value)}
            disabled={submitted}
            rows={3}
          />
        </div>
      )}

      {/* Show correct answer after verify */}
      {hasVerify && !verifyResult && q.answer && (
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

