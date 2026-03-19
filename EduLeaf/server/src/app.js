require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');

const examRoutes      = require('./routes/exams');
const favoriteRoutes  = require('./routes/favorites');
const { sequelize }   = require('./models');

const app  = express();
const PORT = process.env.PORT || 3002;

// ── Middleware ────────────────────────────────────────────────────────────────
app.use(cors({
  origin: 'http://localhost:3000',
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  allowedHeaders: ['Content-Type', 'Authorization'],
}));

app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// ── Static uploads ────────────────────────────────────────────────────────────
app.use('/uploads', express.static(path.join(__dirname, '../uploads')));

// ── Routes ────────────────────────────────────────────────────────────────────
app.use('/api/exams',     examRoutes);
app.use('/api/favorites', favoriteRoutes);

// ── Health check ──────────────────────────────────────────────────────────────
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', service: 'EduLeaf API', timestamp: new Date().toISOString() });
});

// ── 404 handler ───────────────────────────────────────────────────────────────
app.use((req, res) => {
  res.status(404).json({ error: 'Route not found' });
});

// ── Global error handler ──────────────────────────────────────────────────────
app.use((err, req, res, _next) => {
  console.error(err.stack);
  res.status(err.status || 500).json({ error: err.message || 'Internal server error' });
});

// ── Start ─────────────────────────────────────────────────────────────────────
async function start() {
  try {
    // await sequelize.query('PRAGMA foreign_keys = OFF');
    await sequelize.sync({ alter: true });
    // await sequelize.query('PRAGMA foreign_keys = ON');
    console.log('✅ Database synced');
    app.listen(PORT, () => {
      console.log(`🚀 EduLeaf API running on http://localhost:${PORT}`);
    });
  } catch (err) {
    console.error('❌ Failed to start:', err);
    process.exit(1);
  }
}

start();
