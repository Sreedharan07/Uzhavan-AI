import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Eye, Menu, X, ArrowUpRight } from 'lucide-react';
import { type Language } from '../utils/translations';

interface NavbarProps {
  elderlyMode: boolean;
  setElderlyMode: (val: boolean) => void;
  lang: Language;
  setLang: (l: Language) => void;
  isOnline: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  elderlyMode,
  setElderlyMode,
  lang,
  setLang,
  isOnline,
}) => {
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Center navigation items specified in Section 1
  const navLinks = [
    { path: '/assistant', label_ta: 'உதவியாளர்', label_en: 'Assistant' },
    { path: '/agriculture', label_ta: 'வேளாண்மை', label_en: 'Agriculture' },
    { path: '/schemes', label_ta: 'திட்டங்கள்', label_en: 'Schemes' },
    { path: '/crop-diagnosis', label_ta: 'பயிர் மருத்துவர்', label_en: 'Crop Doctor' },
    { path: '/admin', label_ta: 'ஆராய்ச்சி / AI', label_en: 'About' },
  ];

  return (
    <header className="sticky top-3 z-50 px-4 sm:px-6 max-w-[1240px] mx-auto w-full">
      <div className="glass-nav-pill bg-[#05060f]/92 px-4 sm:px-6 py-2.5 flex items-center justify-between gap-4 transition-all duration-300 shadow-[0_20px_40px_rgba(0,0,0,0.8)] border border-[#bad7f7]/15">
        
        {/* Left: Geometric Agriculture/Leaf Logo with Subtle AI Node Motif */}
        <Link to="/" className="flex items-center gap-2.5 group select-none shrink-0">
          <div className="relative w-8 h-8 rounded-full bg-[#bad7f7]/[0.06] border border-[#bad7f7]/20 flex items-center justify-center group-hover:border-[#663af3]/60 transition-colors shadow-[0_0_12px_rgba(102,58,243,0.15)]">
            {/* SVG Geometric Leaf + AI Node Motif */}
            <svg
              className="w-4 h-4 text-[#d8ecf8] group-hover:text-[#a284ff] transition-colors"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              {/* Leaf contour */}
              <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z" />
              <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12" />
              {/* AI Node circle inside leaf */}
              <circle cx="12" cy="11" r="1.5" fill="#663af3" stroke="none" />
              <circle cx="17" cy="8" r="1" fill="#bad7f7" stroke="none" />
            </svg>
          </div>

          <div className="flex flex-col text-left">
            <div className="flex items-center gap-1.5">
              <span className="font-bold text-sm tracking-wider text-white">UZHAVAN AI</span>
              <span className="text-[10px] font-mono px-1.5 py-0.2 rounded-sm bg-[#663af3]/30 text-[#d8ecf8] border border-[#663af3]/40">
                v2.0
              </span>
            </div>
          </div>
        </Link>

        {/* Center Navigation: Minimal Pill Links */}
        <nav className="hidden md:flex items-center gap-1 bg-[#bad7f7]/[0.02] border border-[#bad7f7]/[0.08] px-2 py-1 rounded-full">
          {navLinks.map((item) => {
            const active = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`px-3.5 py-1.5 rounded-full text-xs font-medium transition-all duration-200 ${
                  active
                    ? 'bg-white/10 text-white font-semibold shadow-[inset_0_1px_0_rgba(255,255,255,0.15)]'
                    : 'text-[#9da7ba] hover:text-[#d8ecf8] hover:bg-white/[0.04]'
                }`}
              >
                {lang === 'ta' ? item.label_ta : item.label_en}
              </Link>
            );
          })}
        </nav>

        {/* Right: Language Selector + Get Started Action */}
        <div className="flex items-center gap-2 sm:gap-2.5 shrink-0">
          {/* Segmented Language Selector Pill */}
          <div className="flex items-center p-0.5 rounded-full bg-[#bad7f7]/[0.08] border border-[#bad7f7]/20 text-xs select-none">
            <button
              type="button"
              onClick={() => setLang('ta')}
              aria-label="Switch to Tamil"
              className={`px-2.5 sm:px-3 py-1 rounded-full text-xs font-bold transition-all duration-200 cursor-pointer ${
                lang === 'ta'
                  ? 'bg-[#663af3] text-white shadow-[0_0_12px_rgba(102,58,243,0.5)]'
                  : 'text-[#9da7ba] hover:text-[#d8ecf8]'
              }`}
            >
              தமிழ்
            </button>
            <button
              type="button"
              onClick={() => setLang('en')}
              aria-label="Switch to English"
              className={`px-2.5 sm:px-3 py-1 rounded-full text-xs font-bold transition-all duration-200 cursor-pointer ${
                lang === 'en'
                  ? 'bg-[#663af3] text-white shadow-[0_0_12px_rgba(102,58,243,0.5)]'
                  : 'text-[#9da7ba] hover:text-[#d8ecf8]'
              }`}
            >
              English
            </button>
          </div>

          {/* Accessibility / Elderly Mode Pill */}
          <button
            onClick={() => setElderlyMode(!elderlyMode)}
            aria-label="Toggle Elderly Mode"
            title="Toggle Large Font Accessibility"
            className={`hidden sm:flex items-center gap-1 px-2.5 py-1.5 rounded-full text-[11px] font-medium transition-all border ${
              elderlyMode
                ? 'bg-amber-400/20 text-amber-200 border-amber-400/40'
                : 'bg-transparent text-[#9da7ba] border-[#bad7f7]/10 hover:border-[#bad7f7]/25'
            }`}
          >
            <Eye className="w-3 h-3" />
            <span>{elderlyMode ? (lang === 'ta' ? 'பெரிய எழுத்து' : 'Large Font') : 'A+'}</span>
          </button>

          {/* Primary "Get Started" Pill Button */}
          <Link
            to="/assistant"
            className="btn-primary-violet px-3.5 sm:px-4 py-1.5 text-xs font-bold flex items-center gap-1 text-white select-none hover:scale-105"
          >
            <span>{lang === 'ta' ? 'தொடங்குக' : 'Get Started'}</span>
            <ArrowUpRight className="w-3.5 h-3.5" />
          </Link>

          {/* Mobile Hamburger Toggle */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-1.5 text-[#9da7ba] hover:text-white rounded-lg"
            aria-label="Toggle menu"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer Menu */}
      {mobileMenuOpen && (
        <div className="md:hidden mt-2 glass-plate p-4 space-y-3 text-left animate-in slide-in-from-top-2 duration-200">
          {/* Mobile Language Toggle */}
          <div className="flex items-center justify-between pb-2 border-b border-white/10">
            <span className="text-xs text-[#9da7ba]">{lang === 'ta' ? 'மொழி / Language' : 'Language'}</span>
            <div className="flex items-center p-0.5 rounded-full bg-[#bad7f7]/[0.08] border border-[#bad7f7]/20 text-xs">
              <button
                type="button"
                onClick={() => setLang('ta')}
                className={`px-3 py-1 rounded-full text-xs font-bold transition-all ${
                  lang === 'ta' ? 'bg-[#663af3] text-white' : 'text-[#9da7ba]'
                }`}
              >
                தமிழ்
              </button>
              <button
                type="button"
                onClick={() => setLang('en')}
                className={`px-3 py-1 rounded-full text-xs font-bold transition-all ${
                  lang === 'en' ? 'bg-[#663af3] text-white' : 'text-[#9da7ba]'
                }`}
              >
                English
              </button>
            </div>
          </div>

          {navLinks.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              onClick={() => setMobileMenuOpen(false)}
              className="block px-3 py-2 rounded-xl text-sm font-medium text-[#d1e4fa] hover:bg-white/[0.06] transition-colors"
            >
              {lang === 'ta' ? item.label_ta : item.label_en}
            </Link>
          ))}
          <div className="pt-2 border-t border-white/10 flex justify-between items-center text-xs text-[#9da7ba]">
            <span>{isOnline ? (lang === 'ta' ? '● ஆன்லைன் (தயார்)' : '● Online (Ready)') : (lang === 'ta' ? '○ ஆஃப்லைன்' : '○ Offline Mode')}</span>
            <button
              onClick={() => {
                setElderlyMode(!elderlyMode);
                setMobileMenuOpen(false);
              }}
              className="text-[#a284ff] font-semibold"
            >
              {elderlyMode ? (lang === 'ta' ? 'இயல்பு எழுத்து' : 'Disable Large Font') : (lang === 'ta' ? 'பெரிய எழுத்து' : 'Enable Large Font')}
            </button>
          </div>
        </div>
      )}
    </header>
  );
};
