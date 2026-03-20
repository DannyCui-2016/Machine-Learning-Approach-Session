'use client';

import Link from 'next/link';
import { useLanguage } from '../../contexts/LanguageContext';
import styles from './Footer.module.css';

export default function Footer() {
  const { t } = useLanguage();

  return (
    <footer className={styles.footer}>
      <div className={styles.inner}>
        <div className={styles.brand}>
          <span className={styles.logo}><img width={25} height={25} src="/logo.svg" alt="Logo" /> KnowGrow</span>
          <p className={styles.tagline}>
            {t('home.hero_sub')}
          </p>
        </div>
        <div className={styles.links}>
          <div className={styles.col}>
            <h4 className={styles.colTitle}>{t('nav.exams')}</h4>
            <Link href="/exams/spanish" className={styles.link}>{t('exam.spanish')}</Link>
            <Link href="/exams/german" className={styles.link}>{t('exam.german')}</Link>
            <Link href="/exams/english-pte" className={styles.link}>{t('exam.english_pte')}</Link>
          </div>
          <div className={styles.col}>
            <h4 className={styles.colTitle}>{t('nav.math')}</h4>
            <Link href="/math" className={styles.link}>{t('math.primary')}</Link>
            <Link href="/math" className={styles.link}>{t('math.intermediate')}</Link>
            <Link href="/math" className={styles.link}>{t('math.secondary')}</Link>
          </div>
          <div className={styles.col}>
            <h4 className={styles.colTitle}>{t('nav.home')}</h4>
            <Link href="/favorites" className={styles.link}>{t('nav.favorites')}</Link>
            <Link href="/history" className={styles.link}>{t('nav.history')}</Link>
          </div>
        </div>
      </div>
      <div className={styles.bottom}>
        <p className={styles.copy}>© 2026 KnowGrow. Made for learners everywhere.</p>
      </div>
    </footer>
  );
}
