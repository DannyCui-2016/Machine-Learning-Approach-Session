'use client';

import Link from 'next/link';
import { useLanguage } from '../../contexts/LanguageContext';
import styles from './page.module.css';

const SUBJECTS = [
  {
    id: 'spanish',
    flag: '🇪🇸',
    color: '#E91E63',
    bgPale: '#FCE4EC',
    border: '#F48FB1',
  },
  {
    id: 'german',
    flag: '🇩🇪',
    color: '#1565C0',
    bgPale: '#E3F2FD',
    border: '#90CAF9',
  },
  {
    id: 'english-pte',
    flag: '🇬🇧',
    color: '#6A1B9A',
    bgPale: '#F3E5F5',
    border: '#CE93D8',
  },
];

export default function ExamsPage() {
  const { t } = useLanguage();

  return (
    <div className={styles.page}>
      {/* Header */}
      <section className={styles.header}>
        <div className={styles.headerBlob} />
        <div className="container">
          <span className="section-tag">🎓 {t('exam.title')}</span>
          <h1 className={styles.title}>{t('exam.choose_subject')}</h1>
          <p className={styles.subtitle}>{t('exam.subject_sub')}</p>
        </div>
      </section>

      {/* Subject Cards */}
      <section className={styles.subjects}>
        <div className="container">
          <div className={styles.grid}>
            {SUBJECTS.map((s) => (
              <Link
                key={s.id}
                href={`/exams/${s.id}`}
                className={styles.subjectCard}
                style={{
                  '--card-color': s.color,
                  '--card-bg': s.bgPale,
                  '--card-border': s.border,
                }}
              >
                <span className={styles.flag}>{s.flag}</span>
                <h3 className={styles.subjectName} style={{ color: s.color }}>
                  {t(`exam.${s.id.replace('-', '_')}`)}
                </h3>
                <p className={styles.subjectDesc}>
                  {t('exam.subject_sub')}
                </p>
                <span className={styles.arrow} style={{ color: s.color }}>→</span>
              </Link>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
