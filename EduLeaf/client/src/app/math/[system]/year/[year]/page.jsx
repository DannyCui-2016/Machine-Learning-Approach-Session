'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useLanguage } from '../../../../../contexts/LanguageContext';
import { getCurriculumData } from '../../../../../data/mathCurriculum';
import styles from './page.module.css';

const SYSTEM_COLORS = {
  ncea: { light: '#E8F5E9', main: '#4CAF50', dark: '#2E7D32', pale: '#F1F8E9' },
  cambridge: { light: '#E3F2FD', main: '#1976D2', dark: '#0D47A1', pale: '#EBF5FB' },
  ib: { light: '#F3E5F5', main: '#7B1FA2', dark: '#4A148C', pale: '#F4ECF7' },
};

export default function YearSyllabusPage({ params }) {
  const { system, year } = params;
  const { t } = useLanguage();
  const [activeTab, setActiveTab] = useState('overview');

  const data = getCurriculumData(system, year);
  const colors = SYSTEM_COLORS[system] || SYSTEM_COLORS['ncea'];

  if (!data) {
    return (
      <div className={styles.page} style={{ '--sys-color': colors.main, '--sys-color-light': colors.light, '--sys-color-dark': colors.dark, '--sys-color-pale': colors.pale }}>
        <section className={styles.header}>
          <div className="container">
            <Link href={`/math/${system}`} className="btn btn-ghost btn-sm" style={{ marginBottom: '1rem', display: 'inline-flex', alignItems: 'center', gap: '4px', textDecoration: 'none' }}>
              ← {t('common.back')}
            </Link>
          </div>
        </section>
        <div className="container">
          <div className={styles.emptyState}>
            <div className={styles.emptyEmoji}>🚧</div>
            <h2>{t('math.coming_soon')}</h2>
            <p>{t('math.coming_desc')}</p>
            <Link href={`/math/${system}`} className="btn btn-primary btn-sm" style={{ marginTop: '1rem' }}>
              {t('common.back')}
            </Link>
          </div>
        </div>
      </div>
    );
  }

  const tabs = [
    { id: 'overview', icon: '🎯', labelKey: 'math.tab_overview', defaultLabel: 'Roadmap & Goals' },
    { id: 'topics', icon: '📚', labelKey: 'math.tab_topics', defaultLabel: 'Topics & Resources' },
    { id: 'papers', icon: '📝', labelKey: 'math.tab_papers', defaultLabel: 'Past Papers' },
  ];

  return (
    <div className={styles.page} style={{ '--sys-color': colors.main, '--sys-color-light': colors.light, '--sys-color-dark': colors.dark, '--sys-color-pale': colors.pale }}>
      {/* ── Header ── */}
      <section className={styles.header}>
        <div className="container">
          <Link href={`/math/${system}`} className="btn btn-ghost btn-sm" style={{ marginBottom: '1rem', display: 'inline-flex', alignItems: 'center', gap: '4px', textDecoration: 'none' }}>
            ← {t('common.back')}
          </Link>
          <span className="section-tag" style={{ display: 'inline-block', marginBottom: '1rem', margin: '0 auto' }}>
            Year {year} • {data.subtitle}
          </span>
          <h1 className={styles.title}>{data.title}</h1>
          <p className={styles.desc}>{data.description}</p>
        </div>
      </section>

      {/* ── Main Content ── */}
      <section className="container">
        <div className={styles.layout}>
          {/* Sidebar Tabs */}
          <aside className={styles.sidebar}>
            {tabs.map(tab => (
              <button
                key={tab.id}
                className={`${styles.tabBtn} ${activeTab === tab.id ? styles.tabActive : ''}`}
                onClick={() => setActiveTab(tab.id)}
              >
                <span className={styles.tabIcon}>{tab.icon}</span>
                <span>{t(tab.labelKey) !== tab.labelKey ? t(tab.labelKey) : tab.defaultLabel}</span>
              </button>
            ))}
          </aside>

          {/* Current Tab Content */}
          <main className={styles.content}>
            
            {/* Overview Tab */}
            {activeTab === 'overview' && (
              <div className={styles.sectionPanel}>
                <h2 className={styles.sectionTitle}>🎯 Learning Goals</h2>
                <div className={styles.goalsList}>
                  {data.goals.map((goal, idx) => (
                    <div key={idx} className={styles.goalItem}>
                      <span className={styles.goalIcon}>✓</span>
                      <span>{goal}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Topics & Resources Tab */}
            {activeTab === 'topics' && (
              <div className={styles.sectionPanel}>
                <h2 className={styles.sectionTitle}>📚 Core Topics & External Resources</h2>
                
                {data.topics.map((topic) => (
                  <div key={topic.id} className={styles.topicCard}>
                    <div className={styles.topicHeader}>
                      {topic.title}
                    </div>
                    <div>
                      {topic.items.map((item, idx) => (
                        <div key={idx} className={styles.topicItem}>
                          <div className={styles.itemHeader}>
                            <h3 className={styles.itemName}>{item.name}</h3>
                          </div>
                          <div className={styles.itemConcepts}>
                            {item.concepts.map((concept, cIdx) => (
                              <span key={cIdx} className={styles.conceptTag}>{concept}</span>
                            ))}
                          </div>
                          <div className={styles.itemResources}>
                            {item.tutorialLink && (
                              <a href={item.tutorialLink} target="_blank" rel="noopener noreferrer" className={`btn btn-secondary ${styles.resBtn}`}>
                                📺 ExamSolutions Video
                              </a>
                            )}
                            {item.practiceLink && (
                              <a href={item.practiceLink} target="_blank" rel="noopener noreferrer" className={`btn btn-outline ${styles.resBtn}`} style={{ borderColor: 'var(--sys-color)', color: 'var(--sys-color)' }}>
                                ✍️ PMT Practice qs
                              </a>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Past Papers General Tab */}
            {activeTab === 'papers' && (
              <div className={styles.sectionPanel}>
                <h2 className={styles.sectionTitle}>📝 Past Papers</h2>
                {(!data.pastPapers || data.pastPapers.length === 0) ? (
                  <p style={{ color: 'var(--color-text-muted)' }}>No past papers linked yet.</p>
                ) : (
                  <div className={styles.paperGrid}>
                    {data.pastPapers.map((paper, idx) => (
                      <div key={idx} className={styles.paperCard}>
                        <h3 className={styles.paperTitle}>{paper.year} - {paper.season}</h3>
                        <div className={styles.paperLinks}>
                          {paper.paperUrl && (
                            <a href={paper.paperUrl} target="_blank" rel="noopener noreferrer" className="btn btn-primary btn-sm">
                              📄 View Paper
                            </a>
                          )}
                          {paper.markSchemeUrl && (
                            <a href={paper.markSchemeUrl} target="_blank" rel="noopener noreferrer" className="btn btn-secondary btn-sm">
                              ✅ Mark Scheme
                            </a>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                
                <div style={{ marginTop: '2rem', padding: '1.5rem', background: 'var(--sys-color-light)', borderRadius: 'var(--radius-lg)' }}>
                  <h3 style={{ color: 'var(--sys-color-dark)', marginBottom: '0.5rem' }}>Want to test your skills?</h3>
                  <p style={{ fontSize: '0.9rem', marginBottom: '1rem', color: 'var(--sys-color-dark)' }}>
                    Upload these past papers into our AI Exam Generator or create standard mock exams based on your year group!
                  </p>
                  <Link href="/exams" className="btn btn-primary">
                    Go to Exam Engine →
                  </Link>
                </div>
              </div>
            )}

          </main>
        </div>
      </section>
    </div>
  );
}
