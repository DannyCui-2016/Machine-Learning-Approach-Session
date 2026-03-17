'use client';

import { createContext, useContext, useState, useCallback } from 'react';
import en from '../i18n/en';
import zh from '../i18n/zh';

const translations = { en, zh };

const LanguageContext = createContext({
  lang: 'en',
  t: (key) => key,
  toggleLanguage: () => {},
});

export function LanguageProvider({ children }) {
  const [lang, setLang] = useState('en');

  const toggleLanguage = useCallback(() => {
    setLang((prev) => (prev === 'en' ? 'zh' : 'en'));
  }, []);

  const t = useCallback(
    (keyPath) => {
      const keys = keyPath.split('.');
      let result = translations[lang];
      for (const k of keys) {
        result = result?.[k];
      }
      return result ?? keyPath;
    },
    [lang]
  );

  return (
    <LanguageContext.Provider value={{ lang, t, toggleLanguage }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  return useContext(LanguageContext);
}
