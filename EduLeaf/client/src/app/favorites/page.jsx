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
                {favorites.map((fav) => (
                  <FavoriteCard key={fav.id} fav={fav} onRemove={handleRemove} t={t} />
                ))}
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
    mc:        { bg: '#E8F5E9', color: '#2E7D32', label: t('exam_page.multiple_choice') },
    fill:      { bg: '#E3F2FD', color: '#1565C0', label: t('exam_page.fill_in')         },
    listening: { bg: '#F3E5F5', color: '#6A1B9A', label: t('exam_page.listening')       },
    reading:   { bg: '#FFF8E1', color: '#E65100', label: t('exam_page.reading')         },
    writing:   { bg: '#FCE4EC', color: '#880E4F', label: t('exam_page.writing')         },
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
