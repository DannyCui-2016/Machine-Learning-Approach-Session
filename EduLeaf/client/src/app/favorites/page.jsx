'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useLanguage } from '../../contexts/LanguageContext';
import { removeFavorite } from '../../services/examService';
import styles from './page.module.css';

export default function FavoritesPage() {
  const { t } = useLanguage();
  const [favorites, setFavorites] = useState([]);

  useEffect(() => {
    const stored = JSON.parse(localStorage.getItem('eduleaf-favorites') || '[]');
    setFavorites(stored);
  }, []);

  const handleRemove = async (id) => {
    await removeFavorite(id);
    setFavorites((prev) => prev.filter((f) => f.id !== id));
  };

  const handleClearAll = () => {
    localStorage.removeItem('eduleaf-favorites');
    setFavorites([]);
  };

  return (
    <div className={styles.page}>
      {/* Header */}
      <section className={styles.header}>
        <div className={styles.headerBlob} />
        <div className="container">
          <span className="section-tag">⭐ {t('nav.favorites')}</span>
          <h1 className={styles.title}>{t('exam.favorites_title')}</h1>
          <p className={styles.subtitle}>{t('exam.favorites_empty_hint')}</p>
        </div>
      </section>

      <div className="container">
        <div className={styles.content}>
          {favorites.length === 0 ? (
            <div className={styles.empty}>
              <span className={styles.emptyIcon}>⭐</span>
              <h3 className={styles.emptyTitle}>{t('exam.favorites_title')}</h3>
              <p>{t('exam.favorites_empty')}</p>
              <Link href="/exams" className="btn btn-primary btn-lg">
                🎓 {t('home.cta_exam')}
              </Link>
            </div>
          ) : (
            <>
              <div className={styles.toolbar}>
                <p className={styles.count}>
                  {favorites.length} {favorites.length === 1 ? 'question' : 'questions'} saved
                </p>
                <button className="btn btn-ghost btn-sm" onClick={handleClearAll}>
                  🗑️ Clear All
                </button>
              </div>

              <div className={styles.list}>
                {Object.entries(
                  favorites.reduce((acc, fav) => {
                    const key = fav.subject || 'other';
                    if (!acc[key]) acc[key] = [];
                    acc[key].push(fav);
                    return acc;
                  }, {})
                ).map(([subjectKey, items]) => {
                  const subjectMeta = {
                    spanish: { flag: '🇪🇸', label: 'Spanish' },
                    german: { flag: '🇩🇪', label: 'German' },
                    'english-pte': { flag: '🇬🇧', label: 'English PTE' },
                  };
                  const meta = subjectMeta[subjectKey] || { flag: '📚', label: subjectKey };
                  return (
                    <div key={subjectKey} style={{ marginBottom: '2rem' }}>
                      <h2 style={{
                        fontSize: '1.1rem',
                        fontWeight: '700',
                        color: 'var(--color-text-primary)',
                        marginBottom: '1rem',
                        paddingBottom: '0.5rem',
                        borderBottom: '2px solid var(--color-primary-lighter)',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                      }}>
                        {meta.flag} {meta.label}
                        <span style={{
                          fontSize: '0.75rem',
                          fontWeight: '500',
                          color: 'var(--color-text-muted)',
                          background: 'var(--color-bg-subtle)',
                          padding: '2px 8px',
                          borderRadius: '999px',
                        }}>
                          {items.length} questions
                        </span>
                      </h2>
                      {items.map((fav) => (
                        <FavoriteCard key={fav.id} fav={fav} onRemove={handleRemove} t={t} />
                      ))}
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function FavoriteCard({ fav, onRemove, t }) {
  const typeColors = {
    mc: { bg: '#E8F5E9', color: '#2E7D32', label: 'Multiple Choice' },
    fill: { bg: '#E3F2FD', color: '#1565C0', label: 'Fill in the Blank' },
    tf: { bg: '#FFF8E1', color: '#E65100', label: 'True / False' },
    translation: { bg: '#F3E5F5', color: '#6A1B9A', label: 'Translation' },
    listening: { bg: '#FCE4EC', color: '#880E4F', label: 'Listening' },
    reading: { bg: '#FFF3E0', color: '#E65100', label: 'Reading' },
    writing: { bg: '#FCE4EC', color: '#880E4F', label: 'Writing' },
  };
  const meta = typeColors[fav.type] || typeColors.mc;

  return (
    <div className={styles.card}>
      <div className={styles.cardHeader}>
        <span className={styles.typeTag} style={{ background: meta.bg, color: meta.color }}>
          {meta.label}
        </span>
        {fav.savedAt && (
          <span className={styles.savedDate}>
            Saved {new Date(fav.savedAt).toLocaleDateString()}
          </span>
        )}
        <button
          className={styles.removeBtn}
          onClick={() => onRemove(fav.id)}
          title="Remove from favorites"
        >
          ✕
        </button>
      </div>

      {fav.passage && (
        <div className={styles.passage}>{fav.passage.slice(0, 200)}{fav.passage.length > 200 ? '…' : ''}</div>
      )}

      <p className={styles.question}>{fav.question}</p>

      {fav.type === 'mc' && fav.options && (
        <div className={styles.options}>
          {fav.options.map((opt, i) => {
            const letter = String.fromCharCode(65 + i);
            return (
              <span
                key={letter}
                className={`${styles.option} ${letter === fav.answer ? styles.optionCorrect : ''}`}
              >
                <strong>{letter}.</strong> {opt}
              </span>
            );
          })}
        </div>
      )}

      {fav.answer && fav.type !== 'mc' && fav.type !== 'writing' && (
        <div className={styles.answer}>
          💡 {t('exam_page.model_answer')}: <strong>{fav.answer}</strong>
        </div>
      )}

      {fav.points && (
        <div className={styles.points}>{fav.points} pts</div>
      )}
    </div>
  );
}
