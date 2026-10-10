import React from 'react';

export type WaveformState = 'idle' | 'listening' | 'speaking';

interface VoiceWaveformProps {
  state?: WaveformState;
  className?: string;
}

export const VoiceWaveform: React.FC<VoiceWaveformProps> = ({
  state = 'idle',
  className = '',
}) => {
  // 16 bars with synchronized heights based on state
  const barCounts = 18;

  return (
    <div className={`flex items-center justify-center gap-1.5 h-10 px-4 select-none ${className}`}>
      {Array.from({ length: barCounts }).map((_, i) => {
        // Compute heights and delays
        let heightClass = 'h-1.5 opacity-30';
        let animStyle: React.CSSProperties = {};

        if (state === 'idle') {
          // Subtle resting wave
          const centerDist = Math.abs(i - barCounts / 2);
          const baseH = Math.max(3, 10 - centerDist * 1.2);
          heightClass = 'bg-[#c7d3ea]/40';
          animStyle = {
            height: `${baseH}px`,
            transition: 'all 0.4s ease',
          };
        } else if (state === 'listening') {
          // Dynamic pulsing waves
          const delay = (i % 6) * 0.12;
          const minH = 6;
          const maxH = 28;
          heightClass = 'bg-gradient-to-t from-[#663af3] to-[#d8ecf8] shadow-[0_0_8px_rgba(102,58,243,0.5)]';
          animStyle = {
            animation: `waveform-pulse 0.9s ease-in-out infinite alternate`,
            animationDelay: `${delay}s`,
            minHeight: `${minH}px`,
            maxHeight: `${maxH}px`,
          };
        } else if (state === 'speaking') {
          // Speech cadence variation
          const delay = (i % 4) * 0.15;
          heightClass = 'bg-gradient-to-t from-[#663af3] via-[#a284ff] to-[#ffffff] shadow-[0_0_12px_rgba(186,214,247,0.6)]';
          animStyle = {
            animation: `waveform-pulse 0.6s cubic-bezier(0.4, 0, 0.2, 1) infinite alternate`,
            animationDelay: `${delay}s`,
            minHeight: `${4}px`,
            maxHeight: `${32}px`,
          };
        }

        return (
          <span
            key={i}
            className={`w-[3px] rounded-full transition-all duration-300 ${heightClass}`}
            style={animStyle}
          />
        );
      })}

      <style>{`
        @keyframes waveform-pulse {
          0% {
            height: 4px;
            opacity: 0.4;
          }
          50% {
            height: 24px;
            opacity: 0.9;
          }
          100% {
            height: 32px;
            opacity: 1;
          }
        }
      `}</style>
    </div>
  );
};
