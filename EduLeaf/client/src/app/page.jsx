'use client';

import Link from 'next/link';
import { useLanguage } from '../contexts/LanguageContext';
import styles from './page.module.css';

const STATS = [
  { value: '2,400+', keyId: 'stats_exams' },
  { value: '3', keyId: 'stats_subjects' },
  { value: '1,200+', keyId: 'stats_students' },
];

const STEPS = [
  { keyTitle: 'step1_title', keyDesc: 'step1_desc' },
  { keyTitle: 'step2_title', keyDesc: 'step2_desc' },
  { keyTitle: 'step3_title', keyDesc: 'step3_desc' },
];

export default function HomePage() {
  const { t } = useLanguage();

  return (
    <div className={styles.page}>
      {/* ── Hero ── */}
      <section className={styles.hero}>
        <div className={styles.heroContent}>
          <h1 className={styles.heroTitle}>
            {t('home.hero_title')}<br />
            <span className={styles.heroAccent}>{t('home.hero_title_accent')}</span>
          </h1>
          <p className={styles.heroSub}>{t('home.hero_sub')}</p>
          <div className={styles.heroCtas}>
            <Link href="/exams" className="btn btn-primary btn-lg">
              {t('home.cta_exam')}
            </Link>
            <Link href="/math" className="btn btn-secondary btn-lg">
              {t('home.cta_math')}
            </Link>
          </div>
        </div>
      </section>

      {/* ── Modules ── */}
      <section className={styles.modules}>
        <div className="container">
          <div className="text-center">
            <span className="section-tag">{t('home.modules_title')}</span>
            <h2 className="section-title">{t('home.modules_title')}</h2>
            <p className="section-subtitle">{t('home.modules_sub')}</p>
            <div className="divider" />
          </div>
          <div className={styles.moduleGrid}>
            {/* Module 1 */}
            <div className={`${styles.moduleCard} ${styles.moduleCard1}`}>
              <div className={styles.moduleEmoji}>📑</div>
              <span className="badge badge-green">{t('home.module1_tag')}</span>
              <h3 className={styles.moduleTitle}>{t('home.module1_title')}</h3>
              <p className={styles.moduleDesc}>{t('home.module1_desc')}</p>
              <div className={styles.moduleSubjects}>
                <span className={styles.subjectPill}>🇪🇸 Spanish</span>
                <span className={styles.subjectPill}>🇩🇪 German</span>
                <span className={styles.subjectPill}>🇬🇧 PTE</span>
              </div>
              <Link href="/exams" className="btn btn-primary">
                {t('home.cta_exam')} →
              </Link>
            </div>
            {/* Module 2 */}
            <div className={`${styles.moduleCard} ${styles.moduleCard2}`}>
              <div className={styles.moduleEmoji}>📐</div>
              <span className="badge badge-amber">{t('home.module2_tag')}</span>
              <h3 className={styles.moduleTitle}>{t('home.module2_title')}</h3>
              <p className={styles.moduleDesc}>{t('home.module2_desc')}</p>
              <div className={styles.moduleSubjects}>
                <span className={styles.subjectPillAmber}>{t('math.primary')}</span>
                <span className={styles.subjectPillAmber}>{t('math.intermediate')}</span>
                <span className={styles.subjectPillAmber}>{t('math.secondary')}</span>
              </div>
              <Link href="/math" className="btn btn-secondary">
                {t('home.cta_math')} →
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* ── How It Works ── */}
      <section className={styles.howSection}>
        <div className="container">
          <div className="text-center">
            <span className="section-tag">{t('home.how_title')}</span>
            <h2 className="section-title">{t('home.how_title')}</h2>
            <p className="section-subtitle">{t('home.how_sub')}</p>
            <div className="divider" />
          </div>
          <div className={styles.stepsRow}>
            {STEPS.map((step, i) => (
              <div key={i} className={styles.stepItem}>
                <div className={styles.stepNum}>{i + 1}</div>
                <h4 className={styles.stepTitle}>{t(`home.${step.keyTitle}`)}</h4>
                <p className={styles.stepDesc}>{t(`home.${step.keyDesc}`)}</p>
                {i < STEPS.length - 1 && <div className={styles.stepArrow}>→</div>}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA Banner ── */}
      <section className={styles.ctaBanner}>
        <div className="container">
          <div className={styles.ctaInner}>
            <h2 className={styles.ctaTitle}>{t('home.hero_title')}<br />{t('home.hero_title_accent')}</h2>
            <p className={styles.ctaSub}>{t('home.hero_sub')}</p>
            <Link href="/exams" className="btn btn-primary btn-lg">
              {t('home.cta_exam')} <img src="/logo.png" alt="Logo" width={20} height={20} />
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
