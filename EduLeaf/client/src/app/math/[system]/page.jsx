'use client';

import Link from 'next/link';
import { useLanguage } from '../../../contexts/LanguageContext';
import styles from './page.module.css';

const YEAR_GROUPS = [
  {
    groupKey: 'primary',
    icon: '🌱',
    years: [1, 2, 3, 4, 5, 6],
  },
  {
    groupKey: 'intermediate',
    icon: '🌿',
    years: [7, 8],
  },
  {
    groupKey: 'secondary',
    icon: '🌴',
    years: [9, 10, 11, 12, 13],
  },
];

const SYSTEM_COLORS = {
  ncea: {
    primary: '#81C784',
    intermediate: '#4CAF50',
    secondary: '#2E7D32',
  },
  cambridge: {
    primary: '#64B5F6',
    intermediate: '#1E88E5',
    secondary: '#1565C0',
  },
  ib: {
    primary: '#BA68C8',
    intermediate: '#8E24AA',
    secondary: '#6A1B9A',
  },
};

const TOPICS_PREVIEW = {
  1: ['Counting', 'Basic Addition', 'Shapes'],
  2: ['Place Value', 'Subtraction', 'Measurement'],
  3: ['Multiplication', 'Fractions', 'Time'],
  4: ['Division', 'Decimals', 'Geometry'],
  5: ['Large Numbers', 'Percentages', 'Statistics'],
  6: ['Ratios', 'Algebra Intro', 'Area & Perimeter'],
  7: ['Linear Equations', 'Proportion', 'Data Analysis'],
  8: ['Quadratics Intro', 'Probability', 'Pythagoras'],
  9: ['Algebra', 'Trigonometry', 'Coordinate Plane'],
  10: ['Calculus Intro', 'Sequences', 'Matrices'],
  11: ['AS Maths', 'Statistics', 'Number Theory'],
  12: ['Calculus', 'Complex Numbers', 'Proof'],
  13: ['Differentiation', 'Integration', 'Advanced Stats'],
};

export default function SystemMathPage({ params }) {
  const { system } = params;
  const { t } = useLanguage();

  return (
    <div className={styles.page}>
      {/* ── Header ── */}
      <section className={styles.header}>
        <div className={styles.headerBlob1} />
        <div className={styles.headerBlob2} />
        <div className="container">
          <Link href="/math" className="btn btn-ghost btn-sm" style={{ marginBottom: '1rem', display: 'inline-flex', alignItems: 'center', gap: '4px', textDecoration: 'none' }}>
            ← {t('common.back')}
          </Link>
          <span className="section-tag" style={{ display: 'block', marginTop: '1rem' }}>📐 {t('math.title')}</span>
          <h1 className={styles.title}>{t(`math.system_${system}`)}</h1>
          <p className={styles.subtitle}>{t(`math.system_${system}_desc`)}</p>
          <div className={styles.comingSoonBanner}>
            <span className={styles.comingIcon}>🛠️</span>
            <span>{t('math.template_note')}</span>
          </div>
        </div>
      </section>

      {/* ── Year Groups ── */}
      <section className={styles.groupsSection}>
        <div className="container">
          {YEAR_GROUPS.map((group) => {
            const groupColor = SYSTEM_COLORS[system]?.[group.groupKey] || '#4CAF50';
            return (
              <div key={group.groupKey} className={styles.group}>
                <div className={styles.groupHeader}>
                  <span className={styles.groupIcon}>{group.icon}</span>
                  <h2 className={styles.groupTitle} style={{ color: groupColor }}>
                    {t(`math.${group.groupKey}`)}
                  </h2>
                  <div className={styles.groupDivider} style={{ background: groupColor }} />
                </div>
                <div className={styles.yearsGrid}>
                  {group.years.map((year) => (
                    <YearCard
                      key={year}
                      year={year}
                      color={groupColor}
                      topics={TOPICS_PREVIEW[year] || []}
                      t={t}
                    />
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* ── Bottom CTA ── */}
      <section className={styles.ctaSection}>
        <div className="container">
          <div className={styles.ctaCard}>
            <span className={styles.ctaEmoji}>🎓</span>
            <h3 className={styles.ctaTitle}>{t('home.module1_title')}</h3>
            <p className={styles.ctaDesc}>{t('home.module1_desc')}</p>
            <Link href="/exams" className="btn btn-primary btn-lg">
              {t('home.cta_exam')} →
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}

function YearCard({ year, color, topics, t }) {
  return (
    <div className={styles.yearCard} style={{ '--year-color': color }}>
      <div className={styles.yearBadge} style={{ background: color }}>
        {t('math.year')} {year}
      </div>
      <div className={styles.yearTopics}>
        {topics.map((tp) => (
          <span key={tp} className={styles.topicTag}>{tp}</span>
        ))}
      </div>
      <div className={styles.comingSoonTag}>
        <span>⏳</span>
        <span>{t('math.coming_soon')}</span>
      </div>
    </div>
  );
}
