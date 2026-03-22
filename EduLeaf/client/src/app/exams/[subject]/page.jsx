'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { useLanguage } from '../../../contexts/LanguageContext';
import { useDropzone } from 'react-dropzone';
import { generateExamFromFile, generateExamAuto, getHistory, getFavorites, deleteExamRecord, removeFavorite } from '../../../services/examService';
import styles from './page.module.css';

const SUBJECT_META = {
  spanish: { flag: '🇪🇸', color: '#E91E63', key: 'spanish' },
  german: { flag: '🇩🇪', color: '#1565C0', key: 'german' },
  'english-pte': { flag: '🇬🇧', color: '#6A1B9A', key: 'english_pte' },
};

const LEVELS = [
  { id: 'beginner-1', label: 'Beginner Level 1' },
  { id: 'beginner-2', label: 'Beginner Level 2' },
  { id: 'beginner-3', label: 'Beginner Level 3' },
  { id: 'intermediate-1', label: 'Intermediate Level 1' },
  { id: 'intermediate-2', label: 'Intermediate Level 2' },
  { id: 'intermediate-3', label: 'Intermediate Level 3' },
  { id: 'advanced-1', label: 'Advanced 1' },
  { id: 'advanced-2', label: 'Advanced 2' },
  { id: 'advanced-3', label: 'Advanced 3' },
];



export default function SubjectPage({ params }) {
  const { subject } = params;
  const { t } = useLanguage();
  const meta = SUBJECT_META[subject] || SUBJECT_META.spanish;

  const [tab, setTab] = useState('upload'); // 'upload' | 'auto'
  const [selectedLevel, setSelectedLevel] = useState('');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dashTab, setDashTab] = useState('history');
  const [history, setHistory] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);

  useEffect(() => {
    (async () => {
      setLoadingHistory(true);
      try {
        const data = await getHistory(subject);
        setHistory(data);
      } catch (err) {
        console.error(err);
      } finally {
        setLoadingHistory(false);
      }
    })();
  }, [subject]);

  const onDrop = useCallback((accepted) => {
    if (accepted[0]) setUploadedFile(accepted[0]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'image/*': ['.jpg', '.jpeg', '.png', '.webp'],
    },
    maxSize: 20 * 1024 * 1024,
    multiple: false,
  });

  const handleGenerate = async () => {
    setLoading(true);
    try {
      let exam;
      if (tab === 'upload' && uploadedFile) {
        exam = await generateExamFromFile(uploadedFile, subject);
      } else if (tab === 'auto' && selectedLevel) {
        exam = await generateExamAuto(subject, selectedLevel);
      } else {
        return;
      }
      window.location.href = `/exams/${subject}/exam/${exam.id}`;
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteRecord = async (recordId) => {
    await deleteExamRecord(recordId);
    setHistory((prev) => prev.filter((item) => item.id !== recordId));
  };

  const canGenerate =
    (tab === 'upload' && uploadedFile) ||
    (tab === 'auto' && selectedLevel);

  return (
    <div className={styles.page}>
      {/* Subject Header */}
      <section className={styles.header} style={{ '--accent': meta.color }}>
        <div className="container">
          <Link href="/exams" className={styles.back}>← {t('common.back')}</Link>
          <div className={styles.headerContent}>
            <span className={styles.flag}>{meta.flag}</span>
            <div>
              <h1 className={styles.title}>{t(`exam.${meta.key}`)}</h1>
              <p className={styles.subtitle}>{t('exam.subject_sub_2')}</p>
            </div>
          </div>
        </div>
      </section>

      <div className="container">
        <div className={styles.layout}>
          {/* ── Left: Generator ── */}
          <div className={styles.generator}>
            {/* Tab Toggle */}
            <div className={styles.tabRow}>
              <button
                className={`${styles.tab} ${tab === 'upload' ? styles.tabActive : ''}`}
                onClick={() => setTab('upload')}
              >
                📄 {t('exam.generate_from_file')}
              </button>
              <button
                className={`${styles.tab} ${tab === 'auto' ? styles.tabActive : ''}`}
                onClick={() => setTab('auto')}
              >
                🤖 {t('exam.auto_generate')}
              </button>
            </div>

            {/* Upload Tab */}
            {tab === 'upload' && (
              <div className={styles.panel}>
                <h3 className={styles.panelTitle}>{t('exam.upload_title')}</h3>
                <p className={styles.panelDesc}>{t('exam.upload_desc')}</p>

                <div
                  {...getRootProps()}
                  className={`${styles.dropzone} ${isDragActive ? styles.dropzoneActive : ''} ${uploadedFile ? styles.dropzoneDone : ''}`}
                >
                  <input {...getInputProps()} />
                  {uploadedFile ? (
                    <div className={styles.fileReady}>
                      <span className={styles.fileIcon}>✅</span>
                      <span className={styles.fileName}>{uploadedFile.name}</span>
                      <span className={styles.fileSize}>
                        ({(uploadedFile.size / 1024).toFixed(0)} KB)
                      </span>
                      <button
                        className={styles.removeFile}
                        onClick={(e) => { e.stopPropagation(); setUploadedFile(null); }}
                      >✕</button>
                    </div>
                  ) : (
                    <div className={styles.dropPrompt}>
                      <span className={styles.dropIcon}>📂</span>
                      <p className={styles.dropText}>
                        {isDragActive ? 'Drop it here!' : t('exam.upload_desc')}
                      </p>
                      <button className="btn btn-secondary btn-sm">{t('exam.upload_btn')}</button>
                      <small className={styles.dropHint}>{t('exam.upload_hint')}</small>
                    </div>
                  )}
                </div>

                {/* AI Scan info */}
                <div className={styles.aiInfo}>
                  <span>🤖</span>
                  <p>AI Agent will scan your file and extract content to generate a personalized exam with Multiple Choice, Fill-in, Listening, Reading, and Writing sections.</p>
                </div>
              </div>
            )}

            {/* Auto Generate Tab */}
            {tab === 'auto' && (
              <div className={styles.panel}>
                <h3 className={styles.panelTitle}>{t('exam.levels')}</h3>
                <p className={styles.panelDesc}>{t('exam.auto_generate')}</p>
                <div className={styles.levelsGrid}>
                  {LEVELS.map((lv) => {
                    const group = lv.id.startsWith('beginner')
                      ? 'beginner'
                      : lv.id.startsWith('intermediate')
                        ? 'intermediate'
                        : 'advanced';
                    return (
                      <button
                        key={lv.id}
                        className={`${styles.levelBtn} ${selectedLevel === lv.id ? styles.levelBtnActive : ''} ${styles[`level_${group}`]}`}
                        onClick={() => setSelectedLevel(lv.id)}
                      >
                        {lv.label}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Generate Button */}
            <button
              className={`btn btn-primary btn-lg ${styles.generateBtn}`}
              onClick={handleGenerate}
              disabled={!canGenerate || loading}
            >
              {loading ? (
                <><span className="spinner" /> {t('exam.generating')}</>
              ) : (
                <>✨ {t('exam.generate_btn')}</>
              )}
            </button>
          </div>

          {/* ── Right: Dashboard ── */}
          <div className={styles.dashboard}>
            <div className={styles.dashHeader}>
              <h2 className={styles.dashTitle}>{t('exam.dashboard_title')}</h2>
              <div className={styles.dashTabs}>
                <button
                  className={`${styles.dashTab} ${dashTab === 'history' ? styles.dashTabActive : ''}`}
                  onClick={() => setDashTab('history')}
                >
                  📋 {t('exam.history_title')}
                </button>
                <button
                  className={`${styles.dashTab} ${dashTab === 'favorites' ? styles.dashTabActive : ''}`}
                  onClick={() => setDashTab('favorites')}
                >
                  ⭐ {t('exam.favorites_title')}
                </button>
              </div>
            </div>

            {dashTab === 'history' && (
              <div className={styles.historyList}>
                {loadingHistory ? (
                  <div className={styles.empty}>{t('common.loading')}</div>
                ) : history.length === 0 ? (
                  <div className={styles.empty}>{t('exam.history_empty')}</div>
                ) : (
                  Object.entries(
                    history.reduce((acc, item) => {
                      const title = item.exam?.title || 'Exam';
                      if (!acc[title]) acc[title] = [];
                      acc[title].push(item);
                      return acc;
                    }, {})
                  ).map(([title, items]) => (
                    <div key={title} style={{ marginBottom: '1.5rem' }}>
                      <h3 style={{
                        fontSize: '0.9rem',
                        fontWeight: '600',
                        marginBottom: '0.75rem',
                        color: 'var(--text-main)',
                        borderBottom: '1px solid var(--border-color)',
                        paddingBottom: '0.5rem'
                      }}>
                        {title}
                      </h3>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                        {items.slice(0, 5).map((item, idx) => (
                          <div key={item.id} className={styles.historyRow}>
                            <span style={{ fontSize: '0.75rem', fontWeight: '700', color: 'var(--color-primary-deep)', minWidth: '24px' }}>
                              #{idx + 1}
                            </span>
                            <span style={{ fontSize: '0.72rem', color: 'var(--color-text-muted)', whiteSpace: 'nowrap', minWidth: '80px' }}>
                              {new Date(item.createdAt).toLocaleDateString([], { month: '2-digit', day: '2-digit' })} {new Date(item.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                            </span>
                            <div style={{ flex: 1, height: '6px', background: 'var(--color-border-light)', borderRadius: '999px', overflow: 'hidden', minWidth: '40px' }}>
                              <div style={{ height: '100%', width: `${item.score}%`, background: 'var(--color-primary-grad)', borderRadius: '999px' }} />
                            </div>
                            <span style={{ fontSize: '0.72rem', fontWeight: '700', color: 'var(--color-primary-deep)', whiteSpace: 'nowrap', minWidth: '36px' }}>
                              {item.score}/{item.total}
                            </span>
                            {item.timeElapsed > 0 && (
                              <span style={{ fontSize: '0.72rem', color: 'var(--color-text-muted)', whiteSpace: 'nowrap' }}>
                                ⏱ {Math.floor(item.timeElapsed / 60)}:{String(item.timeElapsed % 60).padStart(2, '0')}
                              </span>
                            )}
                            <div style={{ display: 'flex', gap: '2px', flexShrink: 0 }}>
                              <Link
                                href={`/exams/${subject}/exam/${item.examId || item.exam?.id}?recordId=${item.id}`}
                                className="btn btn-ghost btn-sm"
                                style={{ whiteSpace: 'nowrap' }}
                              >
                                {t('exam.review_btn')}
                              </Link>
                              <button
                                onClick={() => handleDeleteRecord(item.id)}
                                className="btn btn-ghost btn-sm"
                                style={{ color: 'var(--color-error)', padding: '6px 8px' }}
                              >
                                🗑️
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))
                )}
              </div>
            )}

            {dashTab === 'favorites' && (
              <div style={{ overflowY: 'auto', flex: 1, padding: 'var(--space-md)' }}>
                <FavoritesTab key={dashTab} t={t} subject={subject} />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function FavoritesTab({ t, subject }) {
  const [favorites, setFavorites] = useState([]);
  const [pendingRemove, setPendingRemove] = useState({});

  useEffect(() => {
    (async () => {
      const data = await getFavorites(subject);
      setFavorites(data);
    })();
  }, [subject]);

  const handleRemove = (id) => {
    const timer = setTimeout(async () => {
      await removeFavorite(id);
      setFavorites((prev) => prev.filter((f) => f.id !== id));
      setPendingRemove((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
    }, 5000);
    setPendingRemove((prev) => ({ ...prev, [id]: timer }));
  };

  const handleUndo = (id) => {
    clearTimeout(pendingRemove[id]);
    setPendingRemove((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
  };

  if (favorites.length === 0) {
    return (
      <div className={styles.empty}>
        <span style={{ fontSize: '2.5rem' }}>⭐</span>
        <p>{t('exam.favorites_empty')}</p>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      {favorites.map((fav) => {
        const isPending = !!pendingRemove[fav.id];
        return (
          <div key={fav.id} className={styles.historyItem} style={{
            opacity: isPending ? 0.5 : 1,
            transition: 'opacity 0.3s',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.5rem' }}>
              <p style={{ fontSize: '0.85rem', fontWeight: '600', flex: 1 }}>
                {fav.question}
              </p>
              {isPending ? (
                <button
                  onClick={() => handleUndo(fav.id)}
                  title="Undo"
                  style={{
                    background: 'none',
                    border: '1px solid var(--color-border-light)',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '0.75rem',
                    color: 'var(--color-primary)',
                    padding: '2px 8px',
                    whiteSpace: 'nowrap',
                    flexShrink: 0,
                  }}
                >
                  ↩ Undo
                </button>
              ) : (
                <button
                  onClick={() => handleRemove(fav.id)}
                  title="Got it, remove"
                  style={{
                    background: 'none',
                    border: '1px solid var(--color-border-light)',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '0.75rem',
                    color: 'var(--color-text-muted)',
                    padding: '2px 8px',
                    whiteSpace: 'nowrap',
                    flexShrink: 0,
                  }}
                >
                  ✓ Got it
                </button>
              )}
            </div>
            {fav.type === 'mc' && fav.options && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', marginTop: '0.25rem' }}>
                {fav.options.map((opt, i) => {
                  const letter = String.fromCharCode(65 + i);
                  const isCorrect = letter === fav.answer;
                  return (
                    <span key={letter} style={{
                      fontSize: '0.8rem',
                      color: isCorrect ? '#2E7D32' : 'var(--color-text-sub)',
                      fontWeight: isCorrect ? '600' : '400',
                    }}>
                      {letter}. {opt.replace(/^[A-D][.)]\s*/i, '')} {isCorrect ? '✅' : ''}
                    </span>
                  );
                })}
              </div>
            )}
            {fav.type !== 'mc' && fav.answer && (
              <p style={{ fontSize: '0.8rem', color: 'var(--color-text-sub)' }}>
                💡 {fav.answer}
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}
