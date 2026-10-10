import React from 'react';
import { Link } from 'react-router-dom';
import {
  Mic,
  Sprout,
  ShieldCheck,
  Zap,
  ArrowRight,
  Sparkles,
  Activity,
} from 'lucide-react';
import { Hero } from '../components/hero/Hero';
import { GlassSurface } from '../components/hero/GlassSurface';
import { type Language, translations } from '../utils/translations';

interface HomeProps {
  lang: Language;
}

export const Home: React.FC<HomeProps> = ({ lang }) => {
  const t = translations[lang].home;

  const coreModules = [
    {
      to: '/assistant',
      icon: Mic,
      title: lang === 'ta' ? 'குரல் வழி உதவியாளர்' : 'Dialect Voice Assistant',
      badge: lang === 'ta' ? 'நேரடி ASR' : 'Live Speech AI',
      desc:
        lang === 'ta'
          ? 'கொங்கு, மதுரை, நெல்லை, டெல்டா, சென்னை வட்டார வழக்குகளை துல்லியமாக உணரும் வாய்ஸ் AI.'
          : 'Understands Kongu, Madurai, Tirunelveli, Delta, and Chennai Tamil dialects with speech diglossia normalization.',
      linkText: lang === 'ta' ? 'உதவியாளரை திறக்க' : 'Launch Assistant',
      accentColor: 'text-[#a284ff]',
      borderHover: 'hover:border-[#663af3]/50',
    },
    {
      to: '/agriculture',
      icon: Zap,
      title: lang === 'ta' ? '50x அதிவேக பயிர் AI' : '50x Fast Crop AI',
      badge: 'XGBoost JIT',
      desc:
        lang === 'ta'
          ? 'Numba SIMD JIT தொகுக்கப்பட்ட மரங்கள் வழியே ~9 மைக்ரோ வினாடிகளில் உடனடி பயிர் பரிந்துரை.'
          : 'High-speed XGBoost crop recommendations evaluated in ~9 microseconds with 11 agronomic features.',
      linkText: lang === 'ta' ? 'பரிந்துரை பெற' : 'Get Recommendations',
      accentColor: 'text-amber-300',
      borderHover: 'hover:border-amber-400/40',
    },
    {
      to: '/crop-diagnosis',
      icon: Sprout,
      title: lang === 'ta' ? 'பயிர் நோய் கண்டறிதல்' : 'Leaf Disease Doctor',
      badge: 'Multimodal Vision',
      desc:
        lang === 'ta'
          ? 'இலை புகைப்படம் + குரல் அறிகுறிகளை இணைக்கும் மல்டிமாடல் AI தொழில்நுட்பம் (+21% மீட்பு).'
          : 'Fuses leaf photos (MobileNetV3) with farmer-spoken symptoms for accurate field disease diagnosis.',
      linkText: lang === 'ta' ? 'நோயறிதல் தொடங்க' : 'Diagnose Leaf Photo',
      accentColor: 'text-cyan-300',
      borderHover: 'hover:border-cyan-400/40',
    },
    {
      to: '/schemes',
      icon: ShieldCheck,
      title: lang === 'ta' ? 'அரசு நலத்திட்டங்கள்' : 'Government Schemes',
      badge: 'Grounded RAG',
      desc:
        lang === 'ta'
          ? 'PM-KISAN, பயிர் காப்பீடு மற்றும் இலவச மின்சாரம் குறித்த அதிகாரப்பூர்வ வழிகாட்டல்கள்.'
          : 'Official eligibility and documents for PM-KISAN, crop insurance, and state agricultural schemes.',
      linkText: lang === 'ta' ? 'திட்டங்களை பார்க்க' : 'Explore Schemes',
      accentColor: 'text-emerald-400',
      borderHover: 'hover:border-emerald-500/40',
    },
  ];

  return (
    <div className="w-full space-y-16 selection:bg-[#663af3]/40 selection:text-white">
      {/* 1. Cinematic AuthKit-Inspired Hero */}
      <Hero lang={lang} />

      {/* 2. Platform Architecture & Core Features Showcase */}
      <div className="max-w-[1200px] mx-auto px-4 sm:px-6 space-y-8">
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-[#bad7f7]/10 pb-5">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#bad7f7]/[0.04] border border-[#bad7f7]/15">
              <Sparkles className="w-3.5 h-3.5 text-[#a284ff]" />
              <span className="font-mono text-[11px] text-[#9da7ba] uppercase tracking-widest font-bold">
                {lang === 'ta' ? 'அமைப்பின் தொழில்நுட்பங்கள்' : 'PLATFORM CAPABILITIES'}
              </span>
            </div>
            <h2 className="text-2xl md:text-3xl font-black text-white tracking-tight">
              {lang === 'ta'
                ? 'விவசாயிகளுக்கான முழுமையான AI சுற்றுச்சூழல்'
                : 'Engineered for Real-World Rural Deployments'}
            </h2>
          </div>

          <Link
            to="/admin"
            className="text-xs text-[#9da7ba] hover:text-[#d8ecf8] flex items-center gap-1.5 font-mono group transition-colors"
          >
            <span>{lang === 'ta' ? 'மாதிரி அளவீடுகள் & AI சோதனைகள்' : 'Inspect Latency & Research Benchmarks'}</span>
            <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform text-[#a284ff]" />
          </Link>
        </div>

        {/* 4 Feature Cards conforming to 16px radius and frosted glass */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {coreModules.map((item, idx) => {
            const Icon = item.icon;
            return (
              <Link key={idx} to={item.to} className="group block">
                <GlassSurface
                  className={`p-6 h-full flex flex-col justify-between space-y-4 border border-[#bad7f7]/10 ${item.borderHover} hover:scale-[1.02] transition-all duration-300`}
                >
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="w-10 h-10 rounded-full bg-[#bad7f7]/[0.05] border border-[#bad7f7]/15 flex items-center justify-center">
                        <Icon className={`w-5 h-5 ${item.accentColor}`} />
                      </div>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-[#bad7f7]/[0.05] border border-[#bad7f7]/15 text-[#9da7ba]">
                        {item.badge}
                      </span>
                    </div>

                    <h3 className="text-lg font-bold text-white group-hover:text-[#d8ecf8] transition-colors">
                      {item.title}
                    </h3>

                    <p className="text-xs text-[#9da7ba] leading-relaxed">
                      {item.desc}
                    </p>
                  </div>

                  <div className="pt-2 flex items-center gap-1 text-xs font-semibold text-[#a284ff] group-hover:text-white transition-colors">
                    <span>{item.linkText}</span>
                    <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
                  </div>
                </GlassSurface>
              </Link>
            );
          })}
        </div>
      </div>

      {/* 3. Interactive Sample Conversation Accordion */}
      <div className="max-w-[1200px] mx-auto px-4 sm:px-6 pb-6">
        <GlassSurface className="p-8 border border-[#bad7f7]/12 space-y-6">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-full bg-[#663af3]/20 border border-[#663af3]/40 flex items-center justify-center text-[#d8ecf8]">
              <Activity className="w-4 h-4 text-[#a284ff]" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-white">
                {lang === 'ta' ? 'வட்டார வழக்கு உரையாடல் மாதிரி' : 'Sample Rural Tamil Dialogue Flow'}
              </h3>
              <p className="text-xs text-[#9da7ba]">
                {lang === 'ta'
                  ? 'கொங்கு தமிழிலிருந்து தூய உழவியல் ஆலோசனை வரை நேரடி பார்வை'
                  : 'Demonstration of dialect understanding and grounded agricultural response'}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded-xl bg-[#05060f]/60 border border-[#bad7f7]/10 space-y-2">
              <span className="text-[10px] font-mono text-[#a284ff] uppercase font-bold">
                👨‍🌾 {t.farmerLabel}
              </span>
              <p className="text-sm font-medium text-[#d8ecf8] leading-relaxed">
                {t.farmerQuery}
              </p>
            </div>

            <div className="p-4 rounded-xl bg-[#663af3]/10 border border-[#663af3]/25 space-y-2">
              <span className="text-[10px] font-mono text-emerald-400 uppercase font-bold">
                🤖 {t.aiLabel}
              </span>
              <p className="text-sm text-[#d8ecf8] leading-relaxed">
                {t.aiResponse}
              </p>
            </div>
          </div>
        </GlassSurface>
      </div>
    </div>
  );
};
