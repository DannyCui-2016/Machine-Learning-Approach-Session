'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { useLanguage } from '../../../contexts/LanguageContext';
import { useDropzone } from 'react-dropzone';
import { generateExamFromFile, generateExamAuto, getHistory, getFavorites } from '../../../services/examService';
import styles from './page.module.css';

const SUBJECT_META = {
  spanish:      { flag: '🇪🇸', color: '#E91E63', key: 'spanish'      },
  german:       { flag: '🇩🇪', color: '#1565C0', key: 'german'       },
  'english-pte':{ flag: '🇬🇧', color: '#6A1B9A', key: 'english_pte'  },
};

const LEVELS = [
  { id: 'beginner-1',     label: 'Beginner Level 1'     },
  { id: 'beginner-2',     label: 'Beginner Level 2'     },
  { id: 'beginner-3',     label: 'Beginner Level 3'     },
  { id: 'intermediate-1', label: 'Intermediate Level 1' },
  { id: 'intermediate-2', label: 'Intermediate Level 2' },
  { id: 'intermediate-3', label: 'Intermediate Level 3' },
  { id: 'advanced-1',     label: 'Advanced 1'           },
  { id: 'advanced-2',     label: 'Advanced 2'           },
  { id: 'advanced-3',     label: 'Advanced 3'           },
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
              <p className={styles.subtitle}>{t('exam.subject_sub')}</p>
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
                  history.map((item) => (
                    <div key={item.id} className={styles.historyItem}>
                      <div className={styles.historyMeta}>
                        <span className={styles.historyTitle}>{item.exam?.title || 'Exam'}</span>
                        <span className={styles.historyDate}>{new Date(item.createdAt).toLocaleDateString()}</span>
                      </div>
                      <div className={styles.historyScore}>
                        <div className={styles.scoreBar}>
                          <div
                            className={styles.scoreBarFill}
                            style={{ width: `${item.score}%` }}
                          />
                        </div>
                        <span className={styles.scoreVal}>{item.score}/{item.total}</span>
                      </div>
                      <div className={styles.historyActions}>
                        <Link
                          href={`/exams/${subject}/exam/${item.examId || item.exam?.id}?recordId=${item.id}`}
                          className="btn btn-ghost btn-sm"
                        >
                          {t('exam.review_btn')}
                        </Link>
                      </div>
                    </div>
                  ))
                )}
              </div>
            )}

            {dashTab === 'favorites' && (
              <div className={styles.empty}>
                <span style={{ fontSize: '2.5rem' }}>⭐</span>
                <p>{t('exam.favorites_empty')}</p>
                <Link href="/favorites" className="btn btn-secondary btn-sm">
                  {t('nav.favorites')}
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
