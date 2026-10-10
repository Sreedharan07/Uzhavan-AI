import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Navbar } from './components/Navbar';
import { Home } from './pages/Home';
import { Assistant } from './pages/Assistant';
import { CropDiagnosis } from './pages/CropDiagnosis';
import { Schemes } from './pages/Schemes';
import { Agriculture } from './pages/Agriculture';
import { History } from './pages/History';
import { Admin } from './pages/Admin';
import { checkModelStatus } from './services/api';
import { type Language } from './utils/translations';

export function App() {
  const [elderlyMode, setElderlyMode] = useState<boolean>(() => {
    return localStorage.getItem('uzhavan_elderly') === 'true';
  });
  const [lang, setLang] = useState<Language>(() => {
    const saved = localStorage.getItem('uzhavan_lang');
    return (saved === 'en' || saved === 'ta') ? saved : 'ta';
  });
  const [isOnline, setIsOnline] = useState(true);

  const handleSetLang = (l: Language) => {
    setLang(l);
    localStorage.setItem('uzhavan_lang', l);
  };

  const handleSetElderlyMode = (val: boolean) => {
    setElderlyMode(val);
    localStorage.setItem('uzhavan_elderly', String(val));
  };

  useEffect(() => {
    checkModelStatus()
      .then(() => setIsOnline(true))
      .catch(() => setIsOnline(false));

    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return (
    <Router>
      <div className={`min-h-screen flex flex-col font-sans ${elderlyMode ? 'elderly-mode' : ''}`}>
        <Navbar
          elderlyMode={elderlyMode}
          setElderlyMode={handleSetElderlyMode}
          lang={lang}
          setLang={handleSetLang}
          isOnline={isOnline}
        />

        <main className="flex-1 pb-16">
          <Routes>
            <Route path="/" element={<Home lang={lang} />} />
            <Route path="/assistant" element={<Assistant elderlyMode={elderlyMode} lang={lang} />} />
            <Route path="/crop-diagnosis" element={<CropDiagnosis lang={lang} />} />
            <Route path="/schemes" element={<Schemes lang={lang} />} />
            <Route path="/agriculture" element={<Agriculture lang={lang} />} />
            <Route path="/history" element={<History lang={lang} />} />
            <Route path="/admin" element={<Admin lang={lang} />} />
          </Routes>
        </main>

        <footer className="glass-nav py-6 border-t border-[#bad7f7]/10 text-center text-xs text-[#9da7ba] space-y-1 mt-auto">
          <p className="font-medium text-[#d8ecf8]">
            {lang === 'ta'
              ? '🌾 உழவன் AI — வட்டார வழக்கு அறிந்த கிராமப்புற தமிழ் குரல் உதவியாளர்'
              : '🌾 UZHAVAN AI — Dialect-Aware Rural Tamil Voice Assistant'}
          </p>
          <p className="text-[11px] text-[#9da7ba]/70">
            {lang === 'ta'
              ? '© 2026 உழவன் AI திட்டம் | தமிழ்நாடு விவசாயிகள் மற்றும் கிராமப்புற மக்களுக்காக உருவாக்கப்பட்டது'
              : '© 2026 UZHAVAN AI Project | Designed for Tamil Nadu Farmers & Rural Communities'}
          </p>
        </footer>
      </div>
    </Router>
  );
}

export default App;
