import React from 'react';
import { Mic } from 'lucide-react';
import { VoiceWaveform, type WaveformState } from './VoiceWaveform';

interface VoiceOrbProps {
  state: WaveformState;
  onClick: () => void;
  lang?: 'ta' | 'en';
}

export const VoiceOrb: React.FC<VoiceOrbProps> = ({
  state,
  onClick,
  lang = 'ta',
}) => {
  const isListening = state === 'listening';
  const isSpeaking = state === 'speaking';

  return (
    <div className="flex flex-col items-center justify-center space-y-4">
      {/* Interactive Microphone Pill with Halo */}
      <div className="relative group cursor-pointer" onClick={onClick}>
        {/* Expanding Atmospheric Halo */}
        <div
          className={`absolute -inset-4 rounded-full transition-all duration-700 pointer-events-none ${
            isListening || isSpeaking
              ? 'bg-gradient-to-r from-[#663af3]/50 via-[#9bb9e8]/30 to-[#663af3]/50 blur-2xl animate-pulse scale-110'
              : 'bg-[#663af3]/25 group-hover:bg-[#663af3]/50 blur-xl scale-100 group-hover:scale-110'
          }`}
        />

        {/* Pulsing Concentric Ring */}
        {(isListening || isSpeaking) && (
          <div className="absolute -inset-2 rounded-full border border-[#663af3]/60 animate-halo pointer-events-none" />
        )}

        {/* The Main Violet Button Pill */}
        <button
          type="button"
          aria-label={isListening ? 'Stop listening' : 'Start voice assistant'}
          className={`relative z-10 btn-primary-violet px-8 py-4 md:px-10 md:py-4.5 rounded-full flex items-center gap-3.5 text-base md:text-lg font-bold tracking-wide select-none cursor-pointer transition-all duration-300 ${
            isListening
              ? 'ring-2 ring-white/60 bg-[#7b52ff] scale-105'
              : 'group-hover:scale-105'
          }`}
        >
          <div
            className={`w-7 h-7 rounded-full flex items-center justify-center transition-transform ${
              isListening ? 'animate-bounce bg-white/20' : 'group-hover:scale-110'
            }`}
          >
            <Mic className={`w-5 h-5 ${isListening ? 'text-white' : 'text-white'}`} />
          </div>

          <span className="text-white font-semibold">
            {isListening
              ? lang === 'ta'
                ? 'கேட்கிறேன்...'
                : 'Listening...'
              : isSpeaking
              ? lang === 'ta'
                ? 'பதிலளிக்கிறேன்...'
                : 'Speaking...'
              : lang === 'ta'
              ? 'பேச தொடங்குங்கள்'
              : 'Start Speaking'}
          </span>
        </button>
      </div>

      {/* Embedded Audio Waveform Bar */}
      <div className="pt-1 transition-opacity duration-300">
        <VoiceWaveform state={state} />
      </div>
    </div>
  );
};
