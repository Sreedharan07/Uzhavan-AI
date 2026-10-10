import React, { useEffect, useState } from 'react';
import { Activity, Cpu, Database, Server } from 'lucide-react';
import { checkModelStatus, type ModelStatus } from '../services/api';
import { type Language, translations } from '../utils/translations';

interface AdminProps {
  lang: Language;
}

export const Admin: React.FC<AdminProps> = ({ lang }) => {
  const t = translations[lang].admin;
  const [status, setStatus] = useState<ModelStatus | null>(null);

  useEffect(() => {
    checkModelStatus().then(setStatus).catch(() => {});
  }, []);

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-8">
      <div className="glass-card rounded-3xl p-8 space-y-4 border border-emerald-500/25">
        <div className="flex items-center gap-3">
          <Activity className="w-8 h-8 text-emerald-400" />
          <div>
            <h2 className="text-2xl font-bold text-white">{t.title}</h2>
            <p className="text-sm text-slate-300 font-medium">{t.subtitle}</p>
          </div>
        </div>
      </div>

      {status && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="glass-card rounded-2xl p-6 space-y-4 border border-emerald-800/60">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <Cpu className="w-5 h-5 text-emerald-400" />
              <span>{t.gpuInfo}</span>
            </h3>

            <div className="space-y-2 text-sm">
              <div className="flex justify-between py-1.5 border-b border-slate-700/60">
                <span className="text-slate-400 font-medium">{t.device}</span>
                <span className="font-mono font-bold text-emerald-400">{status.device}</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-700/60">
                <span className="text-slate-400 font-medium">{t.gpuModel}</span>
                <span className="font-mono font-semibold text-slate-200">{status.gpu_name}</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-700/60">
                <span className="text-slate-400 font-medium">{t.vram}</span>
                <span className="font-mono font-semibold text-slate-200">{status.vram}</span>
              </div>
            </div>
          </div>

          <div className="glass-card rounded-2xl p-6 space-y-4 border border-emerald-800/60">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <Server className="w-5 h-5 text-teal-400" />
              <span>{t.modelStatus}</span>
            </h3>

            <div className="space-y-2 text-sm">
              <div className="flex justify-between py-1.5 border-b border-slate-700/60">
                <span className="text-slate-400 font-medium">{t.whisperAsr}</span>
                <span className={status.asr_loaded ? 'text-emerald-400 font-bold' : 'text-amber-400 font-bold'}>
                  {status.asr_loaded ? t.loaded : t.idle}
                </span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-700/60">
                <span className="text-slate-400 font-medium">{t.dialectCls}</span>
                <span className={status.dialect_loaded ? 'text-emerald-400 font-bold' : 'text-amber-400 font-bold'}>
                  {status.dialect_loaded ? t.loaded : t.idle}
                </span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-700/60">
                <span className="text-slate-400 font-medium">{t.intentCls}</span>
                <span className={status.intent_loaded ? 'text-emerald-400 font-bold' : 'text-amber-400 font-bold'}>
                  {status.intent_loaded ? t.loaded : t.idle}
                </span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-700/60">
                <span className="text-slate-400 font-medium">{t.cropCnn}</span>
                <span className={status.crop_loaded ? 'text-emerald-400 font-bold' : 'text-amber-400 font-bold'}>
                  {status.crop_loaded ? t.loaded : t.idle}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="glass-card rounded-2xl p-6 space-y-4 border border-emerald-800/60">
        <h3 className="text-lg font-bold text-white flex items-center gap-2">
          <Database className="w-5 h-5 text-amber-400" />
          <span>{t.tensorboardTitle}</span>
        </h3>
        <p className="text-sm text-slate-300 font-medium">{t.tensorboardDesc}</p>
        <pre className="bg-emerald-950/80 p-4 rounded-xl text-emerald-400 font-mono text-sm border border-emerald-800/80 overflow-x-auto">
          tensorboard --logdir results/
        </pre>
      </div>
    </div>
  );
};
