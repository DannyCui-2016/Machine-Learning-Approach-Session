'use client';

import Link from 'next/link';
import { useLanguage } from '../../contexts/LanguageContext';
import styles from './page.module.css';

const YEAR_GROUPS = [
  {
    groupKey: 'primary',
    icon: '🌱',
    color: '#4CAF50',
    years: [1, 2, 3, 4, 5, 6],
  },
  {
    groupKey: 'intermediate',
    icon: '🌿',
    color: '#1976D2',
    years: [7, 8],
  },
  {
    groupKey: 'secondary',
    icon: '🌳',
    color: '#7B1FA2',
    years: [9, 10, 11, 12, 13],
  },
];

const TOPICS_PREVIEW = {
  1:  ['Counting', 'Basic Addition', 'Shapes'],
  2:  ['Place Value', 'Subtraction', 'Measurement'],
  3:  ['Multiplication', 'Fractions', 'Time'],
  4:  ['Division', 'Decimals', 'Geometry'],
  5:  ['Large Numbers', 'Percentages', 'Statistics'],
  6:  ['Ratios', 'Algebra Intro', 'Area & Perimeter'],
  7:  ['Linear Equations', 'Proportion', 'Data Analysis'],
  8:  ['Quadratics Intro', 'Probability', 'Pythagoras'],
  9:  ['Algebra', 'Trigonometry', 'Coordinate Plane'],
  10: ['Calculus Intro', 'Sequences', 'Matrices'],
  11: ['AS Maths', 'Statistics', 'Number Theory'],
  12: ['Calculus', 'Complex Numbers', 'Proof'],
  13: ['Differentiation', 'Integration', 'Advanced Stats'],
};

export default function MathPage() {
  const { t } = useLanguage();

  return (
    <div className={styles.page}>
      {/* ── Header ── */}
      <section className={styles.header}>
        <div className={styles.headerBlob1} />
        <div className={styles.headerBlob2} />
        <div className="container">
          <span className="section-tag">📐 {t('math.title')}</span>
          <h1 className={styles.title}>{t('math.subtitle')}</h1>
          <p className={styles.subtitle}>{t('math.desc')}</p>
          <div className={styles.comingSoonBanner}>
            <span className={styles.comingIcon}>🛠️</span>
            <span>{t('math.template_note')}</span>
          </div>
        </div>
      </section>

      {/* ── Year Groups ── */}
      <section className={styles.groupsSection}>
        <div className="container">
          {YEAR_GROUPS.map((group) => (
            <div key={group.groupKey} className={styles.group}>
              <div className={styles.groupHeader}>
                <span className={styles.groupIcon}>{group.icon}</span>
                <h2 className={styles.groupTitle} style={{ color: group.color }}>
                  {t(`math.${group.groupKey}`)}
                </h2>
                <div className={styles.groupDivider} style={{ background: group.color }} />
              </div>
              <div className={styles.yearsGrid}>
                {group.years.map((year) => (
                  <YearCard
                    key={year}
                    year={year}
                    color={group.color}
                    topics={TOPICS_PREVIEW[year] || []}
                    t={t}
                  />
                ))}
              </div>
            </div>
          ))}
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
