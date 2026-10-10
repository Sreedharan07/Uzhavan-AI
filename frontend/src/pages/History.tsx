import React, { useEffect, useState } from 'react';
import { History as HistoryIcon, Clock } from 'lucide-react';
import { getHistory } from '../services/api';
import { type Language, translations } from '../utils/translations';

interface HistoryProps {
  lang: Language;
}

export const History: React.FC<HistoryProps> = ({ lang }) => {
  const t = translations[lang].history;
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getHistory()
      .then((data) => setHistory(data.history))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="max-w-4xl mx-auto px-4 py-8 space-y-8">
      <div className="glass-card rounded-3xl p-8 space-y-4 border border-emerald-500/25">
        <div className="flex items-center gap-3">
          <HistoryIcon className="w-8 h-8 text-emerald-400" />
          <div>
            <h2 className="text-2xl font-bold text-white">{t.title}</h2>
            <p className="text-sm text-slate-300 font-medium">{t.subtitle}</p>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="text-center text-slate-400 py-12 font-medium">{t.loading}</div>
      ) : history.length === 0 ? (
        <div className="glass-card rounded-2xl p-8 text-center text-slate-400 font-medium">
          {t.empty}
        </div>
      ) : (
        <div className="space-y-4">
          {history.map((item, idx) => (
            <div key={idx} className="glass-card rounded-2xl p-6 space-y-3 border border-emerald-800/60 hover:border-emerald-500/50 transition-all">
              <div className="flex items-center justify-between text-xs text-slate-400 border-b border-slate-700/60 pb-2">
                <span className="flex items-center gap-1 font-medium">
                  <Clock className="w-3.5 h-3.5" />
                  {new Date(item.timestamp).toLocaleString(lang === 'ta' ? 'ta-IN' : 'en-US')}
                </span>
                {item.dialect && (
                  <span className="px-2.5 py-0.5 rounded-full bg-emerald-950 text-emerald-300 border border-emerald-700/70 font-semibold text-xs">
                    {item.dialect}
                  </span>
                )}
              </div>

              <div>
                <p className="text-xs text-slate-400 font-bold uppercase">{t.query}</p>
                <p className="text-base font-semibold text-emerald-300 mt-0.5">"{item.query}"</p>
              </div>

              <div>
                <p className="text-xs text-slate-400 font-bold uppercase">{t.response}</p>
                <p className="text-sm text-slate-200 font-medium mt-0.5 whitespace-pre-line">{item.response}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
