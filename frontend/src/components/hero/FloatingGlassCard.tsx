import React from 'react';
import { GlassSurface } from './GlassSurface';

export type CardPosition = 'left' | 'center' | 'right';

interface FloatingGlassCardProps {
  position: CardPosition;
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
}

export const FloatingGlassCard: React.FC<FloatingGlassCardProps> = ({
  position,
  children,
  className = '',
  onClick,
}) => {
  // Determine tilt and floating animation class based on position
  let posClasses = '';
  if (position === 'left') {
    posClasses = 'float-left-card md:-rotate-2 md:translate-y-4 hover:rotate-0 hover:translate-y-2';
  } else if (position === 'right') {
    posClasses = 'float-right-card md:rotate-2 md:translate-y-4 hover:rotate-0 hover:translate-y-2';
  } else {
    posClasses = 'float-center-card z-20 md:-translate-y-2 hover:-translate-y-4 scale-100 md:scale-105';
  }

  return (
    <div
      onClick={onClick}
      className={`transition-transform duration-500 cursor-pointer ${posClasses} ${className}`}
    >
      <GlassSurface
        elevated={position === 'center'}
        glow={position === 'center'}
        className="p-6 md:p-7 w-full select-none"
      >
        {children}
      </GlassSurface>
    </div>
  );
};
