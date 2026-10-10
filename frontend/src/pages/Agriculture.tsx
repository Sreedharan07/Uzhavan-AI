import React, { useState } from 'react';
import { Sprout, BookOpen, Search, Zap, Gauge, Droplets, Calendar, BarChart2, ShieldCheck, ChevronDown, ChevronUp } from 'lucide-react';
import dictionary from '../../../agriculture/tamil_agri_dictionary.json';
import crops from '../../../agriculture/crops.json';
import { type Language, translations } from '../utils/translations';
import {
  recommendCropXGBoost,
  getXGBoostBenchmark,
  type XGBoostCropRecommendation,
  type XGBoostBenchmarkResult,
} from '../services/api';

interface AgricultureProps {
  lang: Language;
}

export const Agriculture: React.FC<AgricultureProps> = ({ lang }) => {
  const t = translations[lang].agriculture;
  const [search, setSearch] = useState('');

  // XGBoost Recommender State
  const [district, setDistrict] = useState('thanjavur');
  const [soilType, setSoilType] = useState('alluvial');
  const [waterSource, setWaterSource] = useState('canal');
  const [month, setMonth] = useState<number>(() => new Date().getMonth() + 1);

  // Advanced Agronomic Parameters
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [nVal, setNVal] = useState<number>(110);
  const [pVal, setPVal] = useState<number>(45);
  const [kVal, setKVal] = useState<number>(50);
  const [phVal, setPhVal] = useState<number>(7.0);

  // Results State
  const [recLoading, setRecLoading] = useState(false);
  const [recResult, setRecResult] = useState<XGBoostCropRecommendation | null>(null);
  const [recError, setRecError] = useState<string | null>(null);

  // Benchmark State
  const [benchLoading, setBenchLoading] = useState(false);
  const [benchResult, setBenchResult] = useState<XGBoostBenchmarkResult | null>(null);

  const dictEntries = Object.entries(dictionary).filter(
    ([ta, en]) => ta.includes(search) || en.toLowerCase().includes(search.toLowerCase())
  );

  const handleRecommend = async () => {
    setRecLoading(true);
    setRecError(null);
    try {
      const res = await recommendCropXGBoost({
        district,
        soil_type: soilType,
        water_source: waterSource,
        month,
        n: showAdvanced ? nVal : undefined,
        p: showAdvanced ? pVal : undefined,
        k: showAdvanced ? kVal : undefined,
        ph: showAdvanced ? phVal : undefined,
      });
      setRecResult(res);
    } catch (err: any) {
      setRecError(err.message || 'Failed to get recommendation');
    } finally {
      setRecLoading(false);
    }
  };

  const handleRunBenchmark = async () => {
    setBenchLoading(true);
    try {
      const bRes = await getXGBoostBenchmark(250);
      setBenchResult(bRes);
    } catch (err: any) {
      console.error('Benchmark failed:', err);
    } finally {
      setBenchLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-10">
      {/* ========================================================= */}
      {/* ⚡ 50x Fast XGBoost Crop Recommender Section */}
      {/* ========================================================= */}
      <div className="glass-card rounded-3xl p-8 space-y-6 border border-emerald-500/30 shadow-xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-80 h-80 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-emerald-800/60 pb-5">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-emerald-600 to-teal-400 flex items-center justify-center shadow-lg shadow-emerald-950/50">
              <Zap className="w-6 h-6 text-white animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-2xl font-black text-white tracking-tight">{t.xgboostTitle}</h2>
                <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-amber-400/20 text-amber-300 border border-amber-400/40">
                  {t.speedupBadge}
                </span>
              </div>
              <p className="text-xs md:text-sm text-slate-300 mt-1 font-medium">{t.xgboostSubtitle}</p>
            </div>
          </div>

          <button
            onClick={handleRunBenchmark}
            disabled={benchLoading}
            className="px-4 py-2.5 rounded-xl bg-emerald-900/50 hover:bg-emerald-800/70 border border-emerald-500/40 text-emerald-200 text-xs font-bold flex items-center gap-2 transition-all self-start md:self-auto hover:scale-105 active:scale-95"
          >
            <Gauge className="w-4 h-4 text-emerald-400" />
            <span>{benchLoading ? 'Benchmarking...' : t.benchmarkBtn}</span>
          </button>
        </div>

        {/* Input Form Controls */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-xs font-bold text-slate-300 mb-1.5">{t.district}</label>
            <select
              value={district}
              onChange={(e) => setDistrict(e.target.value)}
              className="w-full bg-emerald-950/70 border border-emerald-800/80 rounded-xl px-3 py-2.5 text-sm text-white focus:outline-none focus:border-emerald-500"
            >
              <option value="thanjavur">தஞ்சாவூர் (Thanjavur Delta)</option>
              <option value="kongu">கொங்கு / கோயம்புத்தூர் (Kongu / Coimbatore)</option>
              <option value="nellai">திருநெல்வேலி (Tirunelveli)</option>
              <option value="madurai">மதுரை (Madurai)</option>
              <option value="chennai">சென்னை / காஞ்சிபுரம் (Chennai / Kanchi)</option>
              <option value="default">இதர மாவட்டங்கள் (Other Districts)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-bold text-slate-300 mb-1.5">{t.soil}</label>
            <select
              value={soilType}
              onChange={(e) => setSoilType(e.target.value)}
              className="w-full bg-emerald-950/70 border border-emerald-800/80 rounded-xl px-3 py-2.5 text-sm text-white focus:outline-none focus:border-emerald-500"
            >
              <option value="alluvial">வண்டல் மண் (Alluvial)</option>
              <option value="black">கரிசல் மண் (Black Cotton)</option>
              <option value="red">செம்மண் (Red Laterite)</option>
              <option value="sandy">மணல் கலந்த மண் (Sandy Loam)</option>
              <option value="clay">களிமண் (Clay)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-bold text-slate-300 mb-1.5">{t.water}</label>
            <select
              value={waterSource}
              onChange={(e) => setWaterSource(e.target.value)}
              className="w-full bg-emerald-950/70 border border-emerald-800/80 rounded-xl px-3 py-2.5 text-sm text-white focus:outline-none focus:border-emerald-500"
            >
              <option value="canal">கால்வாய் பாசனம் (Canal Irrigation)</option>
              <option value="borewell">ஆழ்துளை / கிணற்று பாசனம் (Borewell / Well)</option>
              <option value="rainfed">மானாவாரி (Rainfed Dryland)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-bold text-slate-300 mb-1.5">{t.month}</label>
            <select
              value={month}
              onChange={(e) => setMonth(Number(e.target.value))}
              className="w-full bg-emerald-950/70 border border-emerald-800/80 rounded-xl px-3 py-2.5 text-sm text-white focus:outline-none focus:border-emerald-500"
            >
              <option value={6}>ஜூன் (குருவை / காரிஃப்)</option>
              <option value={7}>ஜூலை (குருவை)</option>
              <option value={8}>ஆகஸ்ட் (சம்பா ஆரம்பம்)</option>
              <option value={9}>செப்டம்பர் (சம்பா)</option>
              <option value={10}>அக்டோபர் (சம்பா / ராபி)</option>
              <option value={11}>நவம்பர் (வடகிழக்கு பருவமழை)</option>
              <option value={12}>டிசம்பர் (தாளடி)</option>
              <option value={1}>ஜனவரி (நவரை)</option>
              <option value={2}>பிப்ரவரி (நவரை)</option>
              <option value={3}>மார்ச் (கோடைப் பருவம்)</option>
              <option value={4}>ஏப்ரல் (கோடைப் பருவம்)</option>
              <option value={5}>மே (முன் பட்டம்)</option>
            </select>
          </div>
        </div>

        {/* Toggle Advanced Soil Parameters */}
        <div className="pt-1">
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-xs text-emerald-400 hover:text-emerald-300 flex items-center gap-1 font-semibold transition-colors"
          >
            {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            <span>{showAdvanced ? 'Hide Custom N-P-K & pH Sliders' : '⚙️ Custom Soil Nutrients (N-P-K & pH)'}</span>
          </button>

          {showAdvanced && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-3 p-4 rounded-2xl bg-emerald-950/50 border border-emerald-800/50">
              <div>
                <label className="block text-xs text-slate-300 mb-1">Nitrogen (N): <span className="text-emerald-400 font-bold">{nVal} mg/kg</span></label>
                <input type="range" min="10" max="250" value={nVal} onChange={(e) => setNVal(Number(e.target.value))} className="w-full accent-emerald-500" />
              </div>
              <div>
                <label className="block text-xs text-slate-300 mb-1">Phosphorus (P): <span className="text-emerald-400 font-bold">{pVal} mg/kg</span></label>
                <input type="range" min="5" max="120" value={pVal} onChange={(e) => setPVal(Number(e.target.value))} className="w-full accent-emerald-500" />
              </div>
              <div>
                <label className="block text-xs text-slate-300 mb-1">Potassium (K): <span className="text-emerald-400 font-bold">{kVal} mg/kg</span></label>
                <input type="range" min="10" max="220" value={kVal} onChange={(e) => setKVal(Number(e.target.value))} className="w-full accent-emerald-500" />
              </div>
              <div>
                <label className="block text-xs text-slate-300 mb-1">Soil pH: <span className="text-emerald-400 font-bold">{phVal}</span></label>
                <input type="range" min="4.5" max="8.5" step="0.1" value={phVal} onChange={(e) => setPhVal(Number(e.target.value))} className="w-full accent-emerald-500" />
              </div>
            </div>
          )}
        </div>

        {/* Action Button */}
        <div className="flex items-center gap-4 pt-1">
          <button
            onClick={handleRecommend}
            disabled={recLoading}
            className="px-6 py-3 rounded-2xl bg-gradient-to-r from-emerald-600 to-teal-500 hover:from-emerald-500 hover:to-teal-400 text-white font-bold text-sm shadow-lg shadow-emerald-950/50 flex items-center gap-2 transition-all hover:scale-105 active:scale-95 disabled:opacity-50"
          >
            <Zap className="w-4 h-4" />
            <span>{recLoading ? 'Computing In Microseconds...' : t.recommendBtn}</span>
          </button>
        </div>

        {recError && (
          <div className="p-4 rounded-xl bg-red-950/60 border border-red-800/80 text-red-300 text-xs">
            {recError}
          </div>
        )}

        {/* Recommendations Result Display */}
        {recResult && (
          <div className="space-y-4 pt-4 border-t border-emerald-800/60 animate-in fade-in duration-300">
            {/* Speed & Season Header */}
            <div className="flex flex-wrap items-center justify-between gap-3 bg-emerald-950/60 p-4 rounded-2xl border border-emerald-700/60">
              <div className="flex items-center gap-3">
                <Calendar className="w-5 h-5 text-emerald-400" />
                <div>
                  <span className="text-xs text-slate-400 font-medium">{lang === 'ta' ? 'பருவம்' : 'Season'}:</span>
                  <p className="text-sm font-bold text-emerald-300">{recResult.season_ta} ({recResult.season})</p>
                </div>
              </div>

              {/* ⚡ 50x Fast Metric Badge */}
              <div className="flex items-center gap-2 bg-emerald-900/80 border border-emerald-400/40 px-3.5 py-1.5 rounded-xl shadow-xs">
                <Zap className="w-4 h-4 text-amber-300" />
                <span className="text-xs text-white font-semibold">
                  Latency: <span className="text-emerald-300 font-mono font-bold">{recResult.performance?.inference_time_us} µs</span>
                </span>
                <span className="text-xs px-2 py-0.5 rounded-md bg-amber-400/30 text-amber-200 font-bold border border-amber-400/50">
                  {recResult.performance?.speedup_factor}
                </span>
              </div>
            </div>

            {/* Recommended Crops Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {recResult.top_crops.map((crop, idx) => (
                <div
                  key={idx}
                  className="glass-card rounded-2xl p-5 border border-emerald-500/30 hover:border-emerald-400/60 transition-all flex flex-col justify-between space-y-3 relative overflow-hidden"
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-5 h-5 rounded-full bg-emerald-500/20 text-emerald-300 text-xs font-bold flex items-center justify-center">
                          #{crop.rank}
                        </span>
                        <h4 className="text-lg font-black text-emerald-300">{crop.crop}</h4>
                      </div>
                      <p className="text-xs text-slate-400 font-medium pl-6">{crop.crop_en}</p>
                    </div>

                    <span className="px-2.5 py-1 rounded-lg bg-emerald-500/20 text-emerald-300 text-xs font-bold border border-emerald-500/30">
                      {crop.confidence_pct}
                    </span>
                  </div>

                  {/* Confidence Progress Bar */}
                  <div className="w-full bg-emerald-950 rounded-full h-1.5 overflow-hidden">
                    <div
                      className="bg-gradient-to-r from-emerald-500 to-teal-300 h-full rounded-full"
                      style={{ width: `${Math.min(crop.confidence * 100, 100)}%` }}
                    />
                  </div>

                  <div className="space-y-1.5 text-xs text-slate-300 bg-emerald-950/40 p-3 rounded-xl border border-emerald-900/50">
                    <p><span className="text-slate-400 font-semibold">{lang === 'ta' ? 'எதிர்பார்க்கப்படும் மகசூல்:' : 'Est. Yield:'}</span> {crop.yield_per_acre}</p>
                    <p><span className="text-slate-400 font-semibold">{lang === 'ta' ? 'பயிர் காலம்:' : 'Duration:'}</span> {crop.duration_days} {lang === 'ta' ? 'நாட்கள்' : 'days'}</p>
                    <p className="flex items-center gap-1">
                      <Droplets className="w-3.5 h-3.5 text-cyan-400" />
                      <span className="text-slate-400 font-semibold">{lang === 'ta' ? 'நீர் தேவை:' : 'Water Need:'}</span>
                      <span className="capitalize font-bold text-cyan-300">{crop.water_need}</span>
                    </p>
                    <p><span className="text-slate-400 font-semibold">NPK Ratio:</span> <span className="font-mono text-emerald-400">{crop.npk_ratio}</span></p>
                  </div>

                  {crop.tips_ta && (
                    <p className="text-xs text-emerald-200/90 italic bg-emerald-900/30 p-2.5 rounded-lg border border-emerald-800/40">
                      💡 {crop.tips_ta}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Live Benchmark Results Card */}
        {benchResult && (
          <div className="mt-4 p-5 rounded-2xl bg-emerald-950/80 border border-teal-500/40 space-y-4 animate-in slide-in-from-top-2 duration-300">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <BarChart2 className="w-5 h-5 text-teal-400" />
                <h4 className="font-bold text-white text-base">
                  ⚡ Live Latency Benchmark ({benchResult.iterations} iterations)
                </h4>
              </div>
              <span className="px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-300 font-black text-sm border border-emerald-400/50">
                {benchResult.speedup_summary}
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-700/60 space-y-2">
                <span className="text-xs text-slate-400 uppercase tracking-wider font-bold">Standard XGBoost Booster</span>
                <p className="text-2xl font-black text-slate-200 font-mono">{benchResult.standard_xgboost.mean_us} µs <span className="text-xs text-slate-400">({benchResult.standard_xgboost.mean_ms} ms)</span></p>
                <p className="text-xs text-slate-400">P95: {benchResult.standard_xgboost.p95_us} µs | Throughput: {benchResult.standard_xgboost.throughput_qps.toLocaleString()} QPS</p>
              </div>

              <div className="p-4 rounded-xl bg-emerald-900/40 border border-emerald-500/60 space-y-2">
                <span className="text-xs text-emerald-300 uppercase tracking-wider font-bold">50x Fast JIT Flattened Engine</span>
                <p className="text-2xl font-black text-emerald-300 font-mono">{benchResult.fast_jit_xgboost.mean_us} µs <span className="text-xs text-emerald-400">({benchResult.fast_jit_xgboost.mean_ms} ms)</span></p>
                <p className="text-xs text-emerald-300/80">P95: {benchResult.fast_jit_xgboost.p95_us} µs | Throughput: {benchResult.fast_jit_xgboost.throughput_qps.toLocaleString()} QPS</p>
              </div>
            </div>

            <div className="flex items-center gap-2 text-xs text-slate-300 pt-1">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <span>
                Numerical Equivalence: <strong className="text-emerald-300">{benchResult.numerical_equivalence ? '100% Identical Predictions' : 'Close Approximation'}</strong> (Max Abs Diff: {benchResult.max_prediction_diff.toFixed(6)})
              </span>
            </div>
          </div>
        )}
      </div>

      {/* ========================================================= */}
      {/* Tamil Agricultural Dictionary Section */}
      {/* ========================================================= */}
      <div className="glass-card rounded-3xl p-8 space-y-6 border border-emerald-500/25">
        <div className="flex items-center gap-3">
          <Sprout className="w-8 h-8 text-emerald-400" />
          <div>
            <h2 className="text-2xl font-bold text-white">{t.title}</h2>
            <p className="text-sm text-slate-300 font-medium">{t.subtitle}</p>
          </div>
        </div>

        {/* Dictionary Search */}
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

      {/* Dictionary Terms Grid */}
      <div className="space-y-4">
        <h3 className="text-xl font-bold text-white flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-emerald-400" />
          <span>{t.dictTitle} ({dictEntries.length})</span>
        </h3>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {dictEntries.slice(0, 40).map(([ta, en], idx) => (
            <div key={idx} className="glass-card rounded-xl p-4 border border-emerald-800/60 hover:border-emerald-500/50 transition-all">
              <p className="font-bold text-emerald-300 text-base">{lang === 'ta' ? ta : en}</p>
              <p className="text-xs text-slate-300 mt-1 capitalize font-medium">{lang === 'ta' ? en : ta}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Crops Guide */}
      <div className="space-y-4">
        <h3 className="text-xl font-bold text-white flex items-center gap-2">
          <Sprout className="w-5 h-5 text-teal-400" />
          <span>{t.guideTitle}</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {Object.entries(crops).slice(0, 9).map(([cropName, info]: [string, any], idx) => (
            <div key={idx} className="glass-card rounded-2xl p-6 space-y-3 border border-emerald-800/60 hover:border-emerald-500/50 transition-all">
              <div className="flex justify-between items-center">
                <h4 className="text-lg font-bold text-emerald-400">{lang === 'ta' ? cropName : info.english}</h4>
                <span className="text-xs text-slate-400 italic font-medium">{lang === 'ta' ? info.english : cropName}</span>
              </div>
              <p className="text-xs text-slate-200"><span className="font-semibold text-slate-400">{t.seasonLabel}</span> {info.season?.join(', ')}</p>
              <p className="text-xs text-slate-300"><span className="font-semibold text-slate-400">{t.districtsLabel}</span> {info.regions?.join(', ')}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
