import React from 'react';

interface PrimaryButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  icon?: React.ReactNode;
  isListening?: boolean;
  className?: string;
}

export const PrimaryButton: React.FC<PrimaryButtonProps> = ({
  children,
  icon,
  isListening = false,
  className = '',
  ...props
}) => {
  return (
    <div className="relative inline-flex group">
      {/* Outer Halo Glow */}
      <div
        className={`absolute -inset-1 rounded-full opacity-60 group-hover:opacity-90 blur-xl transition-all duration-500 pointer-events-none ${
          isListening
            ? 'bg-gradient-to-r from-[#663af3] via-[#00d2ff] to-[#663af3] opacity-90 animate-pulse'
            : 'bg-[#663af3]/50 group-hover:bg-[#663af3]/80'
        }`}
      />

      {/* Button */}
      <button
        className={`relative z-10 btn-primary-violet px-7 py-3.5 md:px-8 md:py-4 text-sm md:text-base font-bold flex items-center justify-center gap-3 cursor-pointer select-none ${
          isListening ? 'ring-2 ring-white/40 ring-offset-2 ring-offset-[#05060f]' : ''
        } ${className}`}
        {...props}
      >
        {icon && (
          <span
            className={`transition-transform duration-300 ${
              isListening ? 'animate-bounce text-white' : 'group-hover:scale-110'
            }`}
          >
            {icon}
          </span>
        )}
        <span className="tracking-wide text-white">{children}</span>
      </button>
    </div>
  );
};
