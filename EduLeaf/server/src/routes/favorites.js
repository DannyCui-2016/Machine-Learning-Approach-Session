const express = require('express');
const router  = express.Router();
const { Favorite } = require('../models/models');

// ── GET /api/favorites ────────────────────────────────────────────────────────
router.get('/', async (req, res, next) => {
  try {
    const favs = await Favorite.findAll({ order: [['createdAt', 'DESC']] });
    res.json(favs.map((f) => ({ id: f.id, ...JSON.parse(f.questionJson), savedAt: f.createdAt })));
  } catch (err) { next(err); }
});

// ── POST /api/favorites ───────────────────────────────────────────────────────
router.post('/', async (req, res, next) => {
  try {
    const { questionId, question } = req.body;
    if (!questionId) return res.status(400).json({ error: 'questionId required' });
    const [fav, created] = await Favorite.findOrCreate({
      where: { id: questionId },
      defaults: { questionJson: JSON.stringify(question) },
    });
    res.status(created ? 201 : 200).json({ id: fav.id, created });
  } catch (err) { next(err); }
});

// ── DELETE /api/favorites/:id ─────────────────────────────────────────────────
router.delete('/:id', async (req, res, next) => {
  try {
    const deleted = await Favorite.destroy({ where: { id: req.params.id } });
    if (!deleted) return res.status(404).json({ error: 'Favorite not found' });
    res.json({ message: 'Removed from favorites' });
  } catch (err) { next(err); }
});

module.exports = router;
