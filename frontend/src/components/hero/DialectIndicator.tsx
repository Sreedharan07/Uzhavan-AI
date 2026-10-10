import React, { useState, useEffect } from 'react';

interface DialectIndicatorProps {
  dialect?: string;
  confidence?: number;
  isDetecting?: boolean;
  className?: string;
  lang?: 'ta' | 'en';
}

export const DialectIndicator: React.FC<DialectIndicatorProps> = ({
  dialect = 'Kongu Tamil',
  confidence = 98.4,
  isDetecting = false,
  className = '',
  lang = 'ta',
}) => {
  const [displayConf, setDisplayConf] = useState(confidence);

  // Subtle natural fluctuation in live confidence
  useEffect(() => {
    if (!isDetecting) {
      const interval = setInterval(() => {
        const delta = (Math.random() * 0.4 - 0.2);
        setDisplayConf(Number((confidence + delta).toFixed(1)));
      }, 2500);
      return () => clearInterval(interval);
    }
  }, [confidence, isDetecting]);

  const localizedDialect = lang === 'ta'
    ? (dialect === 'Kongu Tamil' ? 'கொங்கு தமிழ்' : dialect === 'Madurai Tamil' ? 'மதுரை தமிழ்' : dialect === 'Nellai Tamil' ? 'நெல்லை தமிழ்' : dialect)
    : dialect;

  return (
    <div
      className={`inline-flex items-center gap-2.5 px-3.5 py-1.5 rounded-full bg-[#080b1e]/90 border border-[#bad7f7]/20 backdrop-blur-xl shadow-[0_10px_25px_rgba(0,0,0,0.5)] text-xs select-none ${className}`}
    >
      {/* Pulsing indicator dot */}
      <span className="relative flex h-2 w-2">
        <span
          className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
            isDetecting ? 'bg-amber-400' : 'bg-[#663af3]'
          }`}
        />
        <span
          className={`relative inline-flex rounded-full h-2 w-2 ${
            isDetecting ? 'bg-amber-400' : 'bg-[#9bb9e8]'
          }`}
        />
      </span>

      {/* Label & Value */}
      <div className="flex items-center gap-1.5 font-mono">
        <span className="text-[10px] tracking-widest uppercase text-[#9da7ba] font-semibold">
          {isDetecting
            ? (lang === 'ta' ? 'கண்டறிகிறது...' : 'DETECTING...')
            : (lang === 'ta' ? 'வட்டார வழக்கு:' : 'DIALECT DETECTED:')}
        </span>
        <span className="text-xs font-bold text-[#d8ecf8]">
          {isDetecting
            ? (lang === 'ta' ? 'குரல் பண்பு ஆய்வு' : 'Analyzing Acoustic Tone')
            : localizedDialect}
        </span>
        {!isDetecting && (
          <span className="text-xs text-[#a284ff] font-bold">
            {displayConf}%
          </span>
        )}
      </div>
    </div>
  );
};
