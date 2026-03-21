const { DataTypes } = require('sequelize');
const { sequelize } = require('./index');

// ── Exam ─────────────────────────────────────────────────────────────────────
const Exam = sequelize.define('Exam', {
  id: { type: DataTypes.UUID, defaultValue: DataTypes.UUIDV4, primaryKey: true },
  title: { type: DataTypes.STRING, allowNull: false },
  subject: { type: DataTypes.STRING, allowNull: false },
  level: { type: DataTypes.STRING },
  source: { type: DataTypes.ENUM('file', 'auto'), defaultValue: 'auto' },
  sectionsJson: { type: DataTypes.TEXT, allowNull: false },
}, {
  tableName: 'exams',
  timestamps: true,
  getterMethods: {
    sections() { return JSON.parse(this.sectionsJson || '{}'); },
  },
});

// ── ExamRecord (attempt) ──────────────────────────────────────────────────────
const ExamRecord = sequelize.define('ExamRecord', {
  id: { type: DataTypes.UUID, defaultValue: DataTypes.UUIDV4, primaryKey: true },
  examId: { type: DataTypes.UUID, allowNull: false },
  score: { type: DataTypes.INTEGER, defaultValue: 0 },
  total: { type: DataTypes.INTEGER, defaultValue: 100 },
  timeElapsed: { type: DataTypes.INTEGER, defaultValue: 0 },
  answersJson: { type: DataTypes.TEXT },
}, {
  tableName: 'exam_records',
  timestamps: true,
});

// ── Favorite ──────────────────────────────────────────────────────────────────
const Favorite = sequelize.define('Favorite', {
  id: { type: DataTypes.STRING, primaryKey: true },
  questionJson: { type: DataTypes.TEXT, allowNull: false },
}, {
  tableName: 'favorites',
  timestamps: true,
});

// ── Associations ─────────────────────────────────────────────────────────────
Exam.hasMany(ExamRecord, { foreignKey: 'examId', as: 'records' });
ExamRecord.belongsTo(Exam, { foreignKey: 'examId', as: 'exam' });

module.exports = { Exam, ExamRecord, Favorite };
