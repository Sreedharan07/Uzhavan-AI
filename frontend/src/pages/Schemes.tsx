import React, { useEffect, useState } from 'react';
import { ShieldAlert, ExternalLink, AlertTriangle, Search } from 'lucide-react';
import { getSchemes, type Scheme } from '../services/api';
import { type Language, translations } from '../utils/translations';

interface SchemesProps {
  lang: Language;
}

export const Schemes: React.FC<SchemesProps> = ({ lang }) => {
  const t = translations[lang].schemes;

  const [schemes, setSchemes] = useState<Scheme[]>([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [disclaimer, setDisclaimer] = useState('');

  useEffect(() => {
    getSchemes()
      .then((data) => {
        setSchemes(data.schemes);
        setDisclaimer(data.disclaimer);
      })
      .finally(() => setLoading(false));
  }, []);

  const filtered = schemes.filter(
    (s) =>
      s.scheme_name_tamil.includes(search) ||
      s.scheme_name_english.toLowerCase().includes(search.toLowerCase()) ||
      s.eligibility.includes(search)
  );

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-8">
      <div className="glass-card rounded-3xl p-8 space-y-6 border border-emerald-500/25">
        <div className="flex items-center gap-3">
          <ShieldAlert className="w-8 h-8 text-amber-400" />
          <div>
            <h2 className="text-2xl font-bold text-white">{t.title}</h2>
            <p className="text-sm text-slate-300 font-medium">{t.subtitle}</p>
          </div>
        </div>

        {/* Global Safety Alert */}
        <div className="p-4 rounded-xl bg-amber-950/40 border border-amber-800/60 text-amber-300 text-xs md:text-sm flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 flex-shrink-0 text-amber-400 mt-0.5" />
          <div>
            <p className="font-bold">{t.safetyNoticeTitle}</p>
            <p className="mt-1 font-medium">
              {lang === 'ta'
                ? (disclaimer || 'இந்த தகவல் பொது வழிகாட்டுதல் மட்டுமே. உத்தியோகபூர்வ இணையதளத்தில் சரிபார்க்கவும்.')
                : 'This information is for general guidance only. Always verify latest details on official government websites.'}
            </p>
          </div>
        </div>

        {/* Search */}
        <div className="relative">
          <Search className="w-5 h-5 absolute left-4 top-3.5 text-slate-400" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder={t.searchPlaceholder}
            className="w-full bg-emerald-950/50 border border-emerald-800/60 rounded-2xl pl-12 pr-4 py-3 text-slate-100 placeholder-slate-400 focus:outline-none focus:border-emerald-500 focus:bg-emerald-950/80 text-sm shadow-xs transition-colors"
          />
        </div>
      </div>

      {/* Schemes Grid */}
      {loading ? (
        <div className="text-center text-slate-400 py-12 font-medium">{t.loading}</div>
      ) : (
        <div className="grid grid-cols-1 gap-6">
          {filtered.map((scheme, idx) => {
            const eligibilityText = lang === 'ta' ? scheme.eligibility : (scheme.eligibility_english || scheme.eligibility);
            const benefitsText = lang === 'ta' ? scheme.benefits : (scheme.benefits_english || scheme.benefits);
            const docsList = lang === 'ta'
              ? scheme.documents_required
              : (scheme.documents_required_english || scheme.documents_required);

            return (
              <div key={idx} className="glass-card rounded-2xl p-6 space-y-4 border border-emerald-800/60 hover:border-emerald-500/50 transition-all">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-slate-700/60 pb-3">
                  <div>
                    <h3 className="text-xl font-bold text-emerald-400">
                      {lang === 'ta' ? scheme.scheme_name_tamil : scheme.scheme_name_english}
                    </h3>
                    <p className="text-xs text-slate-300 font-medium">
                      {lang === 'ta' ? scheme.scheme_name_english : scheme.scheme_name_tamil}
                    </p>
                  </div>
                  <span className="px-3 py-1 rounded-full bg-emerald-950 text-emerald-300 border border-emerald-700/70 text-xs font-semibold self-start md:self-auto">
                    {scheme.region}
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-xs text-slate-400 font-bold uppercase">{t.eligibility}</p>
                    <p className="text-slate-100 font-medium mt-0.5">{eligibilityText}</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-400 font-bold uppercase">{t.benefits}</p>
                    <p className="text-slate-100 font-medium mt-0.5">{benefitsText}</p>
                  </div>
                </div>

                <div className="text-sm">
                  <p className="text-xs text-slate-400 font-bold uppercase">{t.documents}</p>
                  <p className="text-slate-200 font-medium mt-0.5">{docsList.join(', ')}</p>
                </div>

                <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pt-3 border-t border-slate-700/60 text-xs text-slate-400">
                  <span className="font-medium">{t.department} {scheme.department}</span>
                  {scheme.source_url && (
                    <a
                      href={scheme.source_url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-emerald-400 font-bold hover:underline flex items-center gap-1"
                    >
                      <span>{t.officialLink}</span>
                      <ExternalLink className="w-3.5 h-3.5" />
                    </a>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
