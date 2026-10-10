import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Mic,
  Sprout,
  Building2,
  ArrowRight,
  CheckCircle2,
  Volume2,
  RefreshCw,
} from 'lucide-react';
import { VoiceOrb } from './VoiceOrb';
import { VoiceWaveform, type WaveformState } from './VoiceWaveform';
import { DialectIndicator } from './DialectIndicator';
import { FloatingGlassCard } from './FloatingGlassCard';
import { CapabilityStrip } from './CapabilityStrip';
import { type Language } from '../../utils/translations';

interface HeroProps {
  lang: Language;
}

export const Hero: React.FC<HeroProps> = ({ lang }) => {
  // Voice interaction state
  const [waveState, setWaveState] = useState<WaveformState>('idle');
  const [activeStep, setActiveStep] = useState<number>(0); // 0: Idle, 1: Listening, 2: Transcribed, 3: Answered
  const [detectedDialect, setDetectedDialect] = useState('Kongu Tamil');
  const [isSpeakingResponse, setIsSpeakingResponse] = useState(false);

  // Trigger simulated voice interaction when CTA is clicked
  const handleVoiceClick = () => {
    if (activeStep > 0 && activeStep < 3) return;

    if (activeStep === 3) {
      // Reset to idle
      setActiveStep(0);
      setWaveState('idle');
      return;
    }

    // Step 1: Listening mode starts
    setActiveStep(1);
    setWaveState('listening');

    // Step 2: After 2s, speech captured and dialect detected
    setTimeout(() => {
      setActiveStep(2);
      setDetectedDialect('Kongu Tamil');

      // Step 3: After another 1.8s, AI generates response
      setTimeout(() => {
        setActiveStep(3);
        setWaveState('speaking');
        setIsSpeakingResponse(true);

        // Turn waveform back to idle after speaking completes
        setTimeout(() => {
          setWaveState('idle');
          setIsSpeakingResponse(false);
        }, 5000);
      }, 1800);
    }, 2200);
  };

  const sampleFarmerQuery =
    lang === 'ta'
      ? 'எங்க வயல்ல நெல்லுக்கு இலை மஞ்சளா இருக்குங்க, என்ன உரம் போடலாம்?'
      : 'The leaves on our paddy field are turning yellow. What fertilizer should we apply?';

  const sampleAiResponse =
    lang === 'ta'
      ? 'நெல்லில் இலை மஞ்சள் நிறமடைவது நைட்ரஜன் சத்துக்குறைபாடு அல்லது குலை நோயின் அறிகுறியாகும். ஏக்கருக்கு 120:40:40 NPK பரிந்துரைக்கப்படுகிறது. யூரியா 35 கிலோ மற்றும் பொட்டாஷ் இடவும்.'
      : 'Yellowing of paddy leaves indicates nitrogen deficiency or early blast. A balanced 120:40:40 NPK ratio is recommended. Apply 35 kg urea with potash.';

  return (
    <section className="relative min-h-[92vh] flex flex-col justify-between overflow-hidden pt-6 pb-12">
      {/* 1. Atmospheric Blueprint Grid & Illumination */}
      <div className="absolute inset-0 blueprint-grid pointer-events-none z-0" />
      <div className="absolute inset-0 hero-spotlight pointer-events-none z-0" />
      <div className="absolute top-10 left-1/2 -translate-x-1/2 w-[700px] h-[500px] hero-conic pointer-events-none z-0" />

      {/* Main Content Container (Max 1200px) */}
      <div className="relative z-10 max-w-[1200px] mx-auto px-4 sm:px-6 w-full flex flex-col items-center text-center space-y-7 pt-4">
        
        {/* 2. Hero Eyebrow */}
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
          className="flex items-center gap-3 select-none"
        >
          <span className="h-[1px] w-8 md:w-16 bg-gradient-to-r from-transparent to-[#bad7f7]/30" />
          <div className="px-3 py-1 rounded-full bg-[#bad7f7]/[0.04] border border-[#bad7f7]/15 backdrop-blur-md">
            <span className="font-mono text-[11px] md:text-xs tracking-[0.25em] text-[#9da7ba] uppercase font-semibold">
              {lang === 'ta' ? 'AI • தமிழ் • வேளாண்மை' : 'AI • TAMIL • AGRICULTURE'}
            </span>
          </div>
          <span className="h-[1px] w-8 md:w-16 bg-gradient-to-l from-transparent to-[#bad7f7]/30" />
        </motion.div>

        {/* 3. Main Luminous Headline */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
          className="space-y-4 max-w-4xl"
        >
          <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-[68px] font-black tracking-tight leading-[1.12] text-luminous">
            {lang === 'ta' ? (
              <>
                உங்கள் குரலில்
                <br />
                விவசாயத்தின்
                <br />
                அடுத்த தலைமுறை.
              </>
            ) : (
              <>
                Next-Generation
                <br />
                Agriculture
                <br />
                In Your Voice.
              </>
            )}
          </h1>

          <p className="text-sm md:text-base text-[#9da7ba] font-medium tracking-wide max-w-xl mx-auto">
            {lang === 'ta'
              ? 'உழவன் AI கிராமப்புற தமிழ், வேளாண் சொற்கள் மற்றும் வட்டார வழக்குகளை துல்லியமாக புரிந்துகொள்கிறது.'
              : 'Uzhavan AI understands rural Tamil, agricultural terminology and regional dialects.'}
          </p>
        </motion.div>

        {/* 4. Compact Hero Description */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.25, ease: [0.16, 1, 0.3, 1] }}
          className="text-sm md:text-base text-[#c7d3ea] max-w-[620px] leading-relaxed font-normal"
        >
          {lang === 'ta' ? (
            <>
              பேசுங்கள். கேளுங்கள். தெரிந்துகொள்ளுங்கள்.
              <br className="hidden sm:inline" />
              விவசாயம், அரசு திட்டங்கள், வானிலை, சந்தை விலை மற்றும் பயிர் பராமரிப்பு — எல்லாம் உங்கள் குரலில்.
            </>
          ) : (
            <>
              Speak. Ask. Discover.
              <br className="hidden sm:inline" />
              Agriculture, government schemes, weather, mandi prices, and crop care — all in your natural voice.
            </>
          )}
        </motion.p>

        {/* 5. Primary Central Voice CTA */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.35, ease: [0.16, 1, 0.3, 1] }}
          className="pt-2 flex flex-col items-center"
        >
          <VoiceOrb
            state={waveState}
            onClick={handleVoiceClick}
            lang={lang}
          />

          {/* Quick hint for user */}
          <span className="text-[11px] text-[#9da7ba]/80 font-mono mt-2">
            {activeStep === 0
              ? (lang === 'ta' ? 'நேரடி குரல் தொடர்பை உருவகப்படுத்த கிளிக் செய்யவும்' : 'Click to test live voice interaction simulation')
              : activeStep === 3
              ? (lang === 'ta' ? 'மீண்டும் இயக்க கிளிக் செய்யவும்' : 'Click to reset simulation')
              : (lang === 'ta' ? 'ஆடியோ செயலாக்கப்படுகிறது...' : 'Processing audio stream...')}
          </span>
        </motion.div>

        {/* 6. Floating AI Cards Fan Composition */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, delay: 0.45, ease: [0.16, 1, 0.3, 1] }}
          className="w-full max-w-5xl pt-4 pb-2"
        >
          {/* Dialect Status Floating Indicator */}
          <div className="flex justify-center mb-4">
            <DialectIndicator
              dialect={detectedDialect}
              confidence={98.4}
              isDetecting={activeStep === 1}
              lang={lang}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-5 md:gap-4 items-center">
            
            {/* LEFT CARD — AGRICULTURE */}
            <FloatingGlassCard position="left">
              <div className="space-y-3.5 text-left">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="w-7 h-7 rounded-full bg-emerald-500/15 border border-emerald-400/30 flex items-center justify-center text-emerald-400">
                      <Sprout className="w-4 h-4" />
                    </div>
                    <span className="text-xs font-bold uppercase tracking-wider text-[#d8ecf8]">
                      {lang === 'ta' ? '🌱 வேளாண்மை' : '🌱 Agriculture'}
                    </span>
                  </div>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-white/[0.05] text-[#9da7ba]">
                    {lang === 'ta' ? 'பாசன AI' : 'Irrigation AI'}
                  </span>
                </div>

                <div className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.05]">
                  <p className="text-xs text-[#9da7ba] mb-1 font-mono">
                    {lang === 'ta' ? 'விவசாயி கேள்வி:' : 'Farmer Query:'}
                  </p>
                  <p className="text-sm font-semibold text-[#d1e4fa] leading-snug">
                    {lang === 'ta' ? '"நெல்லுக்கு எப்ப தண்ணி விடணும்?"' : '"When should I irrigate paddy?"'}
                  </p>
                </div>

                <div className="h-[1px] w-full bg-gradient-to-r from-transparent via-[#bad7f7]/15 to-transparent my-1" />

                <div className="space-y-1">
                  <p className="text-[11px] font-mono text-[#a284ff] uppercase font-bold">
                    {lang === 'ta' ? 'AI உதவியாளர்:' : 'AI Assistant:'}
                  </p>
                  <p className="text-xs text-[#c7d3ea] leading-relaxed">
                    {lang === 'ta'
                      ? 'நெல் வளர்ச்சி நிலையைப் பொறுத்து பாசன நேரம் மாறும். குருவை பருவத்தில் தண்டு உருளும் தருணத்தில் 2 செ.மீ நீர் தேக்குவது உகந்தது...'
                      : 'Irrigation timing varies by growth stage. During panicle initiation in Kuruvai season, maintaining 2 cm standing water is ideal...'}
                  </p>
                </div>

                <Link
                  to="/agriculture"
                  className="inline-flex items-center gap-1.5 text-xs text-[#d8ecf8] hover:text-[#a284ff] font-semibold pt-1 group transition-colors"
                >
                  <span>{lang === 'ta' ? 'பரிந்துரையை பார்க்க' : 'View advice'}</span>
                  <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
                </Link>
              </div>
            </FloatingGlassCard>

            {/* CENTER CARD — VOICE AI (Main Visual Hero Anchor) */}
            <FloatingGlassCard position="center">
              <div className="space-y-4 text-left">
                <div className="flex items-center justify-between border-b border-[#bad7f7]/10 pb-3">
                  <div className="flex items-center gap-2">
                    <div className="w-7 h-7 rounded-full bg-[#663af3]/25 border border-[#663af3]/50 flex items-center justify-center text-[#d8ecf8]">
                      <Mic className="w-4 h-4 text-[#a284ff]" />
                    </div>
                    <span className="text-xs font-black tracking-wider text-white">
                      🎙 UZHAVAN AI
                    </span>
                  </div>

                  <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-mono bg-[#663af3]/20 border border-[#663af3]/40 text-[#d8ecf8]">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#00e5ff] animate-ping" />
                    {activeStep === 1
                      ? (lang === 'ta' ? 'கேட்கிறது...' : 'Listening...')
                      : activeStep === 2
                      ? (lang === 'ta' ? 'பகுப்பாய்வு...' : 'Analyzing...')
                      : activeStep === 3
                      ? (lang === 'ta' ? 'செயலில் உள்ள பதில்' : 'Active Response')
                      : (lang === 'ta' ? 'குரல் தயார்' : 'Voice Active')}
                  </span>
                </div>

                {/* Animated Waveform Divider */}
                <div className="py-1">
                  <VoiceWaveform state={waveState} />
                </div>

                {/* Live or Sample Transcription */}
                <div className="space-y-2 p-3.5 rounded-xl bg-[#05060f]/60 border border-[#bad7f7]/12">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-mono text-[#a284ff] font-bold uppercase">
                      {lang === 'ta' ? 'கொங்கு தமிழ் • 98% துல்லியம்' : `${detectedDialect} • 98% confidence`}
                    </span>
                    <span className="text-[10px] text-[#9da7ba] font-mono">
                      {lang === 'ta' ? 'நேரடி ASR' : 'ASR Live'}
                    </span>
                  </div>

                  <p className="text-sm md:text-[15px] font-semibold text-white leading-snug">
                    {activeStep >= 2
                      ? `"${sampleFarmerQuery}"`
                      : (lang === 'ta' ? '"எங்க வயல்ல நெல்லுக்கு இலை மஞ்சளா இருக்கு..."' : '"The leaves on our paddy field are turning yellow..."')}
                  </p>
                </div>

                {/* Step 3 AI Output when triggered */}
                <AnimatePresence>
                  {activeStep === 3 && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="space-y-2 p-3.5 rounded-xl bg-[#663af3]/15 border border-[#663af3]/30"
                    >
                      <div className="flex items-center justify-between text-[10px] font-mono">
                        <span className="text-emerald-400 font-bold flex items-center gap-1">
                          <CheckCircle2 className="w-3 h-3" />
                          {lang === 'ta' ? 'நோக்கம்: உரப் பரிந்துரை (96%)' : 'Intent: Fertilizer Advice (96%)'}
                        </span>
                        <span className={`text-[#a284ff] flex items-center gap-1 ${isSpeakingResponse ? 'animate-pulse font-bold' : ''}`}>
                          <Volume2 className="w-3 h-3" /> {lang === 'ta' ? 'குரல் தொகுப்பு' : 'Synthesized'}
                        </span>
                      </div>
                      <p className="text-xs text-[#d8ecf8] leading-relaxed">
                        {sampleAiResponse}
                      </p>
                    </motion.div>
                  )}
                </AnimatePresence>

                <div className="flex items-center justify-between pt-1">
                  <Link
                    to="/assistant"
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[#663af3] hover:bg-[#7b52ff] text-white text-xs font-bold transition-all shadow-md hover:scale-105"
                  >
                    <span>{lang === 'ta' ? 'குரல் உதவியாளரை திறக்க' : 'Launch Voice Assistant'}</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </Link>

                  <button
                    onClick={handleVoiceClick}
                    className="text-xs text-[#9da7ba] hover:text-white flex items-center gap-1 transition-colors"
                  >
                    <RefreshCw className="w-3 h-3" />
                    <span>{lang === 'ta' ? 'மீண்டும் கேட்க' : 'Replay'}</span>
                  </button>
                </div>
              </div>
            </FloatingGlassCard>

            {/* RIGHT CARD — GOVERNMENT SCHEME */}
            <FloatingGlassCard position="right">
              <div className="space-y-3.5 text-left">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="w-7 h-7 rounded-full bg-cyan-500/15 border border-cyan-400/30 flex items-center justify-center text-cyan-400">
                      <Building2 className="w-4 h-4" />
                    </div>
                    <span className="text-xs font-bold uppercase tracking-wider text-[#d8ecf8]">
                      {lang === 'ta' ? '🏛️ அரசு திட்டங்கள்' : '🏛️ Govt Schemes'}
                    </span>
                  </div>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-white/[0.05] text-[#9da7ba]">
                    {lang === 'ta' ? 'மத்திய & தமிழக அரசு' : 'Central & TN'}
                  </span>
                </div>

                <div className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.05]">
                  <p className="text-sm font-bold text-white mb-0.5">
                    {lang === 'ta' ? 'PM-KISAN சம்மான் நிதி' : 'PM-KISAN Samman Nidhi'}
                  </p>
                  <p className="text-[11px] text-[#9da7ba]">
                    {lang === 'ta' ? 'ஆண்டுக்கு ₹6,000 நேரடி உதவித்தொகை' : 'Direct benefit transfer of ₹6,000 / year'}
                  </p>
                </div>

                <div className="space-y-1.5 text-xs text-[#c7d3ea]">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                    <span>{lang === 'ta' ? 'தகுதி: நில உரிமையுள்ள விவசாயிகள்' : 'Eligibility: Landholding farmers'}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                    <span>{lang === 'ta' ? 'பயன்கள்: 3 தவணைகளில் வங்கி கணக்கில்' : 'Benefits: 3 equal instalments in bank account'}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                    <span>{lang === 'ta' ? 'ஆவணங்கள்: ஆதார், பட்டா சிட்டா, வங்கி புத்தகம்' : 'Documents: Aadhaar, Patta Chitta, Bank Passbook'}</span>
                  </div>
                </div>

                <div className="h-[1px] w-full bg-gradient-to-r from-transparent via-[#bad7f7]/15 to-transparent my-1" />

                <Link
                  to="/schemes"
                  className="inline-flex items-center gap-1.5 text-xs text-[#d8ecf8] hover:text-[#a284ff] font-semibold pt-1 group transition-colors"
                >
                  <span>{lang === 'ta' ? 'தகுதியை சரிபார்க்க' : 'Check eligibility'}</span>
                  <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
                </Link>
              </div>
            </FloatingGlassCard>
          </div>
        </motion.div>

        {/* 7. Trust / Capability Strip */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6, ease: 'easeOut' }}
          className="w-full pt-3"
        >
          <CapabilityStrip lang={lang} />
        </motion.div>

        {/* 8. Small Social-Proof / Purpose Statement */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.75 }}
          className="pt-6 pb-2 text-center space-y-2 border-t border-[#bad7f7]/10 w-full max-w-xl"
        >
          <p className="font-mono text-[11px] tracking-[0.2em] text-[#9da7ba] uppercase font-bold">
            {lang === 'ta' ? 'கிராமப்புற தமிழ் சமூகங்களுக்காக உருவாக்கப்பட்டது' : 'BUILT FOR RURAL TAMIL COMMUNITIES'}
          </p>
          <p className="text-xs md:text-sm text-[#c7d3ea]/90 leading-relaxed max-w-lg mx-auto font-normal">
            {lang === 'ta'
              ? 'விவசாயிகள், முதியவர்கள் மற்றும் டிஜிட்டல் தொழில்நுட்பம் எட்டாத மக்களுக்கான பிரத்யேக குரல் வழி AI உதவியாளர்.'
              : 'Voice-first AI designed for farmers, elderly citizens and communities beyond the traditional digital interface.'}
          </p>
        </motion.div>

      </div>
    </section>
  );
};
