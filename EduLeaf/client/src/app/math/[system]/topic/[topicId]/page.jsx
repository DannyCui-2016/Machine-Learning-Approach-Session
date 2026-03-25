'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useLanguage } from '../../../../../contexts/LanguageContext';
import { getTopicById, CAMBRIDGE_TOPIC_PATH } from '../../../../../data/mathCurriculum';
import styles from './page.module.css';

const SYSTEM_COLORS = {
  cambridge: { main: '#1E88E5', dark: '#0D47A1', light: '#E3F2FD' },
  ncea:      { main: '#4CAF50', dark: '#2E7D32', light: '#E8F5E9' },
  ib:        { main: '#7B1FA2', dark: '#4A148C', light: '#F3E5F5' },
};

// Format markdown-ish text (bold **text**)
function FormattedText({ text }) {
  const parts = text.split('**');
  return (
    <span>
      {parts.map((p, i) =>
        i % 2 === 1 ? <strong key={i}>{p}</strong> : <span key={i}>{p}</span>
      )}
    </span>
  );
}

// ── Practice Problem Card ────────────────────────────────────────────────────
function ProblemCard({ problem, idx, color }) {
  const [showAnswer, setShowAnswer] = useState(false);

  return (
    <div className={styles.problemCard}>
      <div className={styles.problemHeader}>
        <span className={styles.problemNum} style={{ background: color }}>Q{idx + 1}</span>
        <p className={styles.problemQ}>{problem.q}</p>
      </div>
      <button
        className={styles.showAnswerBtn}
        style={{ '--btn-color': color }}
        onClick={() => setShowAnswer((v) => !v)}
      >
        {showAnswer ? '▲ Hide Answer' : '▶ Show Answer'}
      </button>
      {showAnswer && (
        <div className={styles.answerBox} style={{ borderColor: color }}>
          <span className={styles.answerLabel}>Answer</span>
          <div className={styles.answerText}>
            {problem.a.split('\n').map((line, i) => (
              <p key={i} style={{ margin: '2px 0' }}>
                <FormattedText text={line} />
              </p>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Difficulty Dots ─────────────────────────────────────────────────────────
function DifficultyDots({ level, color }) {
  const labels = ['', 'Beginner', 'Elementary', 'Intermediate', 'Advanced', 'Expert'];
  return (
    <div className={styles.diffRow}>
      <span className={styles.diffLabel}>Difficulty:</span>
      <span className={styles.diffDots}>
        {[1, 2, 3, 4, 5].map((i) => (
          <span key={i} className={styles.dot} style={{ background: i <= level ? color : '#ddd' }} />
        ))}
      </span>
      <span className={styles.diffText} style={{ color }}>{labels[level]}</span>
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────
export default function TopicDetailPage({ params }) {
  const { system, topicId } = params;
  const { t } = useLanguage();
  const topic = getTopicById(topicId);
  const colors = SYSTEM_COLORS[system] || SYSTEM_COLORS['cambridge'];

  // Build prev / next nav within the flat topic list
  const allTopics = CAMBRIDGE_TOPIC_PATH.flatMap((s) => s.topics.map((tp) => ({ ...tp, stageColor: s.color })));
  const idx = allTopics.findIndex((tp) => tp.id === topicId);
  const prevTopic = idx > 0 ? allTopics[idx - 1] : null;
  const nextTopic = idx < allTopics.length - 1 ? allTopics[idx + 1] : null;

  if (!topic) {
    return (
      <div className={styles.page} style={{ '--c': colors.main }}>
        <div className="container" style={{ padding: '4rem 0', textAlign: 'center' }}>
          <h2>Topic not found</h2>
          <Link href={`/math/${system}`} className="btn btn-primary btn-sm" style={{ marginTop: '1rem' }}>
            ← Back
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.page} style={{ '--c': colors.main, '--c-dark': colors.dark, '--c-light': colors.light }}>
      {/* ── Header ── */}
      <section className={styles.header}>
        <div className="container">
          <div className={styles.breadcrumb}>
            <Link href="/math" className={styles.crumb}>Math Module</Link>
            <span className={styles.crumbSep}>›</span>
            <Link href={`/math/${system}`} className={styles.crumb}>Cambridge International</Link>
            <span className={styles.crumbSep}>›</span>
            <span className={styles.crumbCurrent}>{topic.title}</span>
          </div>
          <h1 className={styles.title}>{topic.title}</h1>
          <p className={styles.stageTag}>{topic.stageTitle}</p>
          <DifficultyDots level={topic.difficulty} color={colors.main} />
        </div>
      </section>

      {/* ── Body ── */}
      <div className="container">
        <div className={styles.body}>

          {/* Left: Key Concepts + External Resources */}
          <aside className={styles.sidebar}>
            <div className={styles.card}>
              <h2 className={styles.cardTitle} style={{ color: colors.dark }}>📌 Key Concepts</h2>
              <ul className={styles.conceptList}>
                {topic.concepts.map((c, i) => (
                  <li key={i} className={styles.conceptItem}>
                    <span className={styles.conceptBullet} style={{ background: colors.main }} />
                    {c}
                  </li>
                ))}
              </ul>
            </div>

            <div className={styles.card}>
              <h2 className={styles.cardTitle} style={{ color: colors.dark }}>🔗 Free Resources</h2>
              <div className={styles.resourceLinks}>
                {topic.tutorialLink && (
                  <a href={topic.tutorialLink} target="_blank" rel="noopener noreferrer"
                    className={styles.resourceBtn} style={{ background: colors.light, color: colors.dark, borderColor: colors.main }}>
                    📺 ExamSolutions Tutorials
                  </a>
                )}
                {topic.practiceLink && (
                  <a href={topic.practiceLink} target="_blank" rel="noopener noreferrer"
                    className={styles.resourceBtn} style={{ background: colors.light, color: colors.dark, borderColor: colors.main }}>
                    ✍️ PMT Practice Questions
                  </a>
                )}
              </div>
            </div>
          </aside>

          {/* Right: Practice Problems */}
          <main className={styles.main}>
            <h2 className={styles.sectionTitle} style={{ color: colors.dark }}>🧮 Practice Problems</h2>
            <p className={styles.sectionHint}>Work through each question, then click "Show Answer" to check your work.</p>

            <div className={styles.problemList}>
              {(topic.practiceProblems || []).map((prob, i) => (
                <ProblemCard key={i} problem={prob} idx={i} color={colors.main} />
              ))}
            </div>

            {/* Nav between topics */}
            <div className={styles.topicNav}>
              {prevTopic ? (
                <Link href={`/math/${system}/topic/${prevTopic.id}`} className={`btn btn-ghost btn-sm ${styles.navBtn}`}>
                  ← {prevTopic.title}
                </Link>
              ) : <div />}
              {nextTopic && (
                <Link href={`/math/${system}/topic/${nextTopic.id}`} className={`btn btn-primary btn-sm ${styles.navBtn}`}
                  style={{ background: colors.main, borderColor: colors.main }}>
                  {nextTopic.title} →
                </Link>
              )}
            </div>
          </main>

        </div>
      </div>
    </div>
  );
}
