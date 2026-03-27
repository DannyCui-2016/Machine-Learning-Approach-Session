'use client';
import { createContext, useContext, useState, useEffect, useCallback } from 'react';

const KEY = 'eduleaf-math-favorites';
const MathFavoritesContext = createContext();

export function MathFavoritesProvider({ children }) {
  const [favorites, setFavorites] = useState([]);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(KEY);
      if (stored) setFavorites(JSON.parse(stored));
    } catch {}
  }, []);

  const persist = (next) => {
    setFavorites(next);
    localStorage.setItem(KEY, JSON.stringify(next));
  };

  const addFavorite = useCallback((item) => {
    setFavorites((prev) => {
      if (prev.some((f) => f.id === item.id)) return prev;
      const next = [...prev, { ...item, savedAt: new Date().toISOString() }];
      localStorage.setItem(KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  const removeFavorite = useCallback((id) => {
    setFavorites((prev) => {
      const next = prev.filter((f) => f.id !== id);
      localStorage.setItem(KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  const toggleFavorite = useCallback((item) => {
    setFavorites((prev) => {
      const exists = prev.some((f) => f.id === item.id);
      const next = exists ? prev.filter((f) => f.id !== item.id) : [...prev, { ...item, savedAt: new Date().toISOString() }];
      localStorage.setItem(KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  const isFavorited = useCallback((id) => favorites.some((f) => f.id === id), [favorites]);

  const clearAll = useCallback(() => persist([]), []);

  return (
    <MathFavoritesContext.Provider value={{ favorites, addFavorite, removeFavorite, toggleFavorite, isFavorited, clearAll, favCount: favorites.length }}>
      {children}
    </MathFavoritesContext.Provider>
  );
}

export const useMathFavorites = () => {
  const ctx = useContext(MathFavoritesContext);
  if (!ctx) throw new Error('useMathFavorites must be used within MathFavoritesProvider');
  return ctx;
};
