import React from 'react';

interface GlassSurfaceProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  elevated?: boolean;
  className?: string;
  glow?: boolean;
}

export const GlassSurface: React.FC<GlassSurfaceProps> = ({
  children,
  elevated = false,
  className = '',
  glow = false,
  ...props
}) => {
  return (
    <div
      className={`relative rounded-[16px] transition-all duration-300 ${
        elevated ? 'glass-plate-elevated' : 'glass-plate'
      } ${glow ? 'shadow-[0_0_35px_rgba(102,58,243,0.18)]' : ''} ${className}`}
      {...props}
    >
      {/* Top subtle highlight edge */}
      <div className="absolute inset-x-4 top-0 h-[1px] bg-gradient-to-r from-transparent via-white/15 to-transparent pointer-events-none" />
      {children}
    </div>
  );
};
