'use client';

import Link from 'next/link';
import { useLanguage } from '../../contexts/LanguageContext';
import styles from './page.module.css';

const SYSTEMS = [
  {
    key: 'ncea',
    icon: '🇳🇿',
    color: '#4CAF50',
    titleKey: 'math.system_ncea',
    descKey: 'math.system_ncea_desc'
  },
  {
    key: 'cambridge',
    icon: '🇬🇧',
    color: '#1976D2',
    titleKey: 'math.system_cambridge',
    descKey: 'math.system_cambridge_desc'
  },
  {
    key: 'ib',
    icon: '🌐',
    color: '#7B1FA2',
    titleKey: 'math.system_ib',
    descKey: 'math.system_ib_desc'
  }
];

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
        </div>
      </section>

      {/* ── Systems Grid ── */}
      <section className={styles.systemsSection}>
        <div className="container">
          <div className={styles.systemsGrid}>
            {SYSTEMS.map((sys) => (
              <Link key={sys.key} href={`/math/${sys.key}`} style={{ textDecoration: 'none' }}>
                <div className={styles.systemCard} style={{ '--sys-color': sys.color }}>
                  <div className={styles.systemIconWrap} style={{ background: sys.color }}>
                    <span className={styles.systemIcon}>{sys.icon}</span>
                  </div>
                  <h2 className={styles.systemTitle}>{t(sys.titleKey)}</h2>
                  <p className={styles.systemDesc}>{t(sys.descKey)}</p>
                  <div className={styles.systemArrow} style={{ color: sys.color }}>
                    Explore →
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
