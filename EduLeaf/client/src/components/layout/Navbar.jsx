'use client';

import Link from 'next/link';
import { useState } from 'react';
import { useLanguage } from '../../contexts/LanguageContext';
import styles from './Navbar.module.css';

export default function Navbar() {
  const { t, toggleLanguage } = useLanguage();
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <nav className={styles.navbar}>
      <div className={styles.inner}>
        {/* Logo */}
        <Link href="/" className={styles.logo}>
          <span className={styles.logoIcon}>🍃</span>
          <span className={styles.logoText}>EduLeaf</span>
        </Link>

        {/* Desktop Nav */}
        <ul className={styles.navLinks}>
          <li><Link href="/" className={styles.navLink}>{t('nav.home')}</Link></li>
          <li><Link href="/exams" className={styles.navLink}>{t('nav.exams')}</Link></li>
          <li><Link href="/math" className={styles.navLink}>{t('nav.math')}</Link></li>
          <li><Link href="/favorites" className={styles.navLink}>{t('nav.favorites')}</Link></li>
        </ul>

        {/* Right actions */}
        <div className={styles.actions}>
          <button className={styles.langBtn} onClick={toggleLanguage}>
            🌐 {t('common.language')}
          </button>
          {/* Hamburger */}
          <button
            className={styles.hamburger}
            onClick={() => setMenuOpen((v) => !v)}
            aria-label="Toggle menu"
          >
            <span className={menuOpen ? styles.barOpen1 : styles.bar} />
            <span className={menuOpen ? styles.barOpen2 : styles.bar} />
            <span className={menuOpen ? styles.barOpen3 : styles.bar} />
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      {menuOpen && (
        <div className={styles.mobileMenu}>
          <Link href="/" className={styles.mobileLink} onClick={() => setMenuOpen(false)}>{t('nav.home')}</Link>
          <Link href="/exams" className={styles.mobileLink} onClick={() => setMenuOpen(false)}>{t('nav.exams')}</Link>
          <Link href="/math" className={styles.mobileLink} onClick={() => setMenuOpen(false)}>{t('nav.math')}</Link>
          <Link href="/favorites" className={styles.mobileLink} onClick={() => setMenuOpen(false)}>{t('nav.favorites')}</Link>
          <button className={styles.mobileLang} onClick={toggleLanguage}>🌐 {t('common.language')}</button>
        </div>
      )}
    </nav>
  );
}
