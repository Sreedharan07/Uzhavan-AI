import React from 'react';
import { Mic, Sprout, Building2, CloudSun, TrendingUp, Zap } from 'lucide-react';

interface CapabilityStripProps {
  lang?: 'ta' | 'en';
  className?: string;
}

export const CapabilityStrip: React.FC<CapabilityStripProps> = ({
  lang = 'ta',
  className = '',
}) => {
  const capabilities = [
    {
      icon: Mic,
      label_en: 'Tamil Dialects',
      label_ta: 'தமிழ் வட்டார வழக்கு',
      highlight_en: '5 Dialects',
      highlight_ta: '5 வட்டார வழக்குகள்',
    },
    {
      icon: Sprout,
      label_en: 'Agriculture AI',
      label_ta: 'வேளாண் AI அறிவு',
      highlight_en: 'TNAU Verified',
      highlight_ta: 'TNAU சரிபார்ப்பு',
    },
    {
      icon: Building2,
      label_en: 'Govt Schemes',
      label_ta: 'அரசு திட்டங்கள்',
      highlight_en: 'PM-KISAN / PMFBY',
      highlight_ta: 'PM-KISAN / PMFBY',
    },
    {
      icon: CloudSun,
      label_en: 'Weather Forecast',
      label_ta: 'வானிலை முன்னறிவிப்பு',
      highlight_en: 'Hyper-local',
      highlight_ta: 'துல்லிய முன்னறிவிப்பு',
    },
    {
      icon: TrendingUp,
      label_en: 'Market Prices',
      label_ta: 'சந்தை விலை நிலவரம்',
      highlight_en: 'Live Mandi',
      highlight_ta: 'நேரடி மண்டி விலை',
    },
    {
      icon: Zap,
      label_en: '50x Fast Crop AI',
      label_ta: '50x அதிவேக பயிர் AI',
      highlight_en: 'XGBoost JIT',
      highlight_ta: 'XGBoost JIT',
    },
  ];

  return (
    <div className={`w-full max-w-5xl mx-auto ${className}`}>
      <div className="flex flex-wrap items-center justify-center gap-3 md:gap-6 py-4 px-3">
        {capabilities.map((item, idx) => {
          const Icon = item.icon;
          return (
            <div
              key={idx}
              className="flex items-center gap-2.5 px-3.5 py-2 rounded-full bg-[#080b1e]/60 border border-[#bad7f7]/10 hover:border-[#bad7f7]/25 transition-all duration-300 hover:bg-[#080b1e]/90 group select-none"
            >
              {/* Circular Frosted Icon Container */}
              <div className="w-7 h-7 rounded-full bg-[#bad7f7]/[0.06] border border-[#bad7f7]/15 flex items-center justify-center text-[#d8ecf8] group-hover:text-[#a284ff] transition-colors">
                <Icon className="w-3.5 h-3.5 stroke-[1.5]" />
              </div>

              {/* Label */}
              <div className="text-left">
                <p className="text-xs font-semibold text-[#d1e4fa] tracking-wide">
                  {lang === 'ta' ? item.label_ta : item.label_en}
                </p>
                <p className="text-[10px] text-[#9da7ba] font-mono">
                  {lang === 'ta' ? item.highlight_ta : item.highlight_en}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
