import React, { useState } from 'react';
import { Upload, Sprout, AlertTriangle, RefreshCw, CheckCircle, Mic, Sparkles, ShieldCheck } from 'lucide-react';
import { analyzeCropImage, analyzeCropMultimodal, type CropAnalysisResponse, type MultimodalCropResponse } from '../services/api';
import { type Language, translations } from '../utils/translations';

interface CropDiagnosisProps {
  lang: Language;
}

export const CropDiagnosis: React.FC<CropDiagnosisProps> = ({ lang }) => {
  const t = translations[lang].crop;

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [symptomText, setSymptomText] = useState<string>('');
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CropAnalysisResponse | null>(null);
  const [multimodalResult, setMultimodalResult] = useState<MultimodalCropResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setMultimodalResult(null);
      setError(null);
    }
  };

  const startVoiceRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks: BlobPart[] = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/wav' });
        setAudioBlob(blob);
        stream.getTracks().forEach((track) => track.stop());
      };

      recorder.start();
      setMediaRecorder(recorder);
      setIsRecording(true);
    } catch (err: any) {
      setError('Microphone access denied: ' + err.message);
    }
  };

  const stopVoiceRecording = () => {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      setIsRecording(false);
    }
  };

  const handleMultimodalConsultation = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    try {
      if (symptomText.trim() || audioBlob) {
        const mmRes = await analyzeCropMultimodal(selectedFile, symptomText, audioBlob || undefined);
        setMultimodalResult(mmRes);
        setResult(null);
      } else {
        const res = await analyzeCropImage(selectedFile);
        setResult(res);
        setMultimodalResult(null);
      }
    } catch (err: any) {
      setError(err.message || 'Image analysis failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8 space-y-8">
      <div className="glass-card rounded-3xl p-8 space-y-6 border border-emerald-500/25">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Sprout className="w-8 h-8 text-emerald-400" />
            <div>
              <h2 className="text-2xl font-bold text-white">{t.title}</h2>
              <p className="text-sm text-slate-300 font-medium">{t.subtitle}</p>
            </div>
          </div>
          <span className="px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-300 text-xs font-semibold border border-emerald-500/30 flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Multimodal Vision + Voice Symptoms</span>
          </span>
        </div>

        {/* Upload Dropzone */}
        <div className="border-2 border-dashed border-emerald-800/60 hover:border-emerald-500/60 rounded-2xl p-8 text-center bg-emerald-950/40 hover:bg-emerald-950/60 transition-colors">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            id="crop-image-input"
            className="hidden"
          />
          <label htmlFor="crop-image-input" className="cursor-pointer flex flex-col items-center gap-4">
            {previewUrl ? (
              <div className="space-y-2">
                <img src={previewUrl} alt="Crop Preview" className="max-h-64 rounded-xl object-contain border border-emerald-700/60 mx-auto shadow-xs" />
                <p className="text-xs text-emerald-400 font-bold">{t.changeFile}</p>
              </div>
            ) : (
              <>
                <div className="w-16 h-16 rounded-full bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
                  <Upload className="w-8 h-8" />
                </div>
                <div>
                  <p className="text-base font-bold text-slate-200">{t.chooseFile}</p>
                  <p className="text-xs text-slate-400 mt-1">{t.fileLimits}</p>
                </div>
              </>
            )}
          </label>
        </div>

        {/* Complementary Voice / Text Symptom Input (Research Contribution) */}
        <div className="bg-emerald-950/50 rounded-2xl p-5 border border-emerald-800/60 space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-semibold text-emerald-300 flex items-center gap-2">
              <span>🌾 {lang === 'ta' ? 'அறிகுறிகள் சொல்லவும் (குரல் / உரை - விருப்பத்தேர்வு)' : 'Farmer Symptoms (Voice or Text - Optional)'}</span>
            </label>
            <span className="text-xs text-slate-400">
              {lang === 'ta' ? 'மங்கலான புகைப்படங்களிலும் துல்லியத்தை உயர்த்தும்' : 'Enhances accuracy for blurry/field photos'}
            </span>
          </div>

          <div className="flex gap-2">
            <input
              type="text"
              value={symptomText}
              onChange={(e) => setSymptomText(e.target.value)}
              placeholder={lang === 'ta' ? 'எ.கா: இலையில் மஞ்சள் புள்ளி இருக்கு, தண்டு வாடி போச்சு...' : 'e.g., yellow spots on leaf, stem rotting...'}
              className="flex-1 bg-slate-900/80 border border-emerald-700/60 rounded-xl px-4 py-2.5 text-sm text-white placeholder-slate-400 focus:outline-hidden focus:border-emerald-400"
            />

            <button
              type="button"
              onClick={isRecording ? stopVoiceRecording : startVoiceRecording}
              className={`px-4 py-2.5 rounded-xl font-medium text-sm flex items-center gap-2 transition-colors ${
                isRecording
                  ? 'bg-red-600 text-white animate-pulse'
                  : audioBlob
                  ? 'bg-emerald-700 text-white'
                  : 'bg-emerald-900/80 hover:bg-emerald-800 text-emerald-300 border border-emerald-700/60'
              }`}
            >
              <Mic className="w-4 h-4" />
              <span>{isRecording ? (lang === 'ta' ? 'நிறுத்து' : 'Stop') : audioBlob ? (lang === 'ta' ? 'பதிவானது' : 'Recorded') : (lang === 'ta' ? 'குரல்' : 'Speak')}</span>
            </button>
          </div>
        </div>

        {selectedFile && (
          <div className="flex justify-center">
            <button
              onClick={handleMultimodalConsultation}
              disabled={loading}
              className="px-8 py-3.5 rounded-2xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-base flex items-center gap-2 shadow-md shadow-emerald-950/50 cursor-pointer disabled:opacity-50"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-5 h-5 animate-spin" />
                  <span>{t.processing}</span>
                </>
              ) : (
                <>
                  <CheckCircle className="w-5 h-5" />
                  <span>{lang === 'ta' ? 'கூட்டு நோய் பகுப்பாய்வு (பார்வை + குரல்)' : 'Multimodal Diagnosis (Vision + Symptoms)'}</span>
                </>
              )}
            </button>
          </div>
        )}

        {error && (
          <div className="p-4 rounded-xl bg-red-950/80 border border-red-700/80 text-red-200 text-sm font-medium">
            {error}
          </div>
        )}
      </div>

      {/* Multimodal Result Display */}
      {multimodalResult && (
        <div className="glass-card rounded-3xl p-8 space-y-6 border border-emerald-500/25">
          <div className="flex items-center justify-between">
            <h3 className="text-xl font-bold text-white flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-emerald-400" />
              <span>{lang === 'ta' ? 'கூட்டு ஆய்வு முடிவுகள் (Multimodal Diagnosis)' : 'Multimodal Diagnosis Results'}</span>
            </h3>
            <span className="px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
              Alignment: {multimodalResult.symptom_alignment}
            </span>
          </div>

          <div className="bg-emerald-950/60 rounded-2xl p-6 border border-emerald-800/60 space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p className="text-xs text-slate-400 font-bold uppercase">{t.cropLabel}</p>
                <p className="text-2xl font-bold text-emerald-400 mt-0.5">{multimodalResult.crop}</p>
              </div>

              <div>
                <p className="text-xs text-slate-400 font-bold uppercase">{t.diseaseLabel}</p>
                <p className="text-2xl font-bold text-amber-400 mt-0.5">
                  {multimodalResult.treatment?.disease_tamil || multimodalResult.disease}
                </p>
                <p className="text-xs text-slate-400">{multimodalResult.disease}</p>
              </div>
            </div>

            {/* Confidence Comparison Bar */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2 border-t border-emerald-900/60">
              <div>
                <div className="flex justify-between text-xs text-slate-300 font-semibold mb-1">
                  <span>Visual Confidence</span>
                  <span>{Math.round(multimodalResult.visual_confidence * 100)}%</span>
                </div>
                <div className="w-full bg-slate-900 h-2.5 rounded-full overflow-hidden border border-slate-700">
                  <div
                    className="bg-blue-500 h-full rounded-full"
                    style={{ width: `${Math.round(multimodalResult.visual_confidence * 100)}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs text-emerald-300 font-bold mb-1">
                  <span>Multimodal Fused Confidence</span>
                  <span>{Math.round(multimodalResult.multimodal_confidence * 100)}%</span>
                </div>
                <div className="w-full bg-slate-900 h-2.5 rounded-full overflow-hidden border border-emerald-700">
                  <div
                    className="bg-emerald-400 h-full rounded-full"
                    style={{ width: `${Math.round(multimodalResult.multimodal_confidence * 100)}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Verified Treatment Recommendation */}
            {multimodalResult.treatment && (
              <div className="mt-4 p-4 rounded-xl bg-slate-900/80 border border-emerald-800 space-y-2">
                <div className="flex items-center gap-2 text-emerald-400 font-bold text-sm">
                  <ShieldCheck className="w-4 h-4" />
                  <span>{lang === 'ta' ? 'சரிபார்க்கப்பட்ட கள சிகிச்சை (TNAU வழிகாட்டுதல்)' : 'Verified Field Treatment (TNAU Agritech)'}</span>
                </div>
                <p className="text-sm text-slate-200 leading-relaxed font-medium">
                  {multimodalResult.treatment.treatment}
                </p>
                {multimodalResult.treatment.symptoms && (
                  <p className="text-xs text-slate-400">
                    <span className="font-semibold text-slate-300">அறிகுறிகள்: </span>
                    {multimodalResult.treatment.symptoms}
                  </p>
                )}
              </div>
            )}
          </div>

          <div className="p-4 rounded-xl bg-amber-950/40 border border-amber-800/60 text-amber-300 text-xs leading-relaxed flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 flex-shrink-0 text-amber-400 mt-0.5" />
            <span>{multimodalResult.disclaimer}</span>
          </div>
        </div>
      )}

      {/* Standard Result Display fallback */}
      {result && (
        <div className="glass-card rounded-3xl p-8 space-y-6 border border-emerald-500/25">
          <h3 className="text-xl font-bold text-white flex items-center gap-2">
            <span>{t.resultsTitle}</span>
          </h3>

          {result.top_prediction && result.top_prediction.crop ? (
            <div className="bg-emerald-950/60 rounded-2xl p-6 border border-emerald-800/60 space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-slate-400 font-bold uppercase">{t.cropLabel}</p>
                  <p className="text-xl font-bold text-emerald-400 mt-0.5">
                    {lang === 'ta' ? (result.top_prediction.crop_tamil || result.top_prediction.crop) : result.top_prediction.crop}
                  </p>
                </div>

                <div>
                  <p className="text-xs text-slate-400 font-bold uppercase">{t.diseaseLabel}</p>
                  <p className="text-xl font-bold text-amber-400 mt-0.5">{result.top_prediction.possible_disease}</p>
                </div>
              </div>

              <div>
                <p className="text-xs text-slate-400 font-bold uppercase">{t.confidenceLabel}</p>
                <div className="w-full bg-emerald-950 h-3 rounded-full overflow-hidden mt-1.5 border border-emerald-800/60">
                  <div
                    className="bg-emerald-500 h-full rounded-full"
                    style={{ width: `${Math.round(result.top_prediction.confidence * 100)}%` }}
                  />
                </div>
                <p className="text-xs text-slate-300 mt-1 font-semibold text-right">
                  {Math.round(result.top_prediction.confidence * 100)}%
                </p>
              </div>

              {/* Top-3 Differential Predictions */}
              {result.predictions && result.predictions.length > 1 && (
                <div className="pt-3 border-t border-emerald-900/60 space-y-2">
                  <p className="text-xs text-slate-400 font-bold uppercase">
                    {lang === 'ta' ? 'சாத்தியமான பிற கணிப்புகள் (Top Differential Diagnoses):' : 'Top Differential Diagnoses:'}
                  </p>
                  <div className="space-y-1.5">
                    {result.predictions.slice(1, 3).map((pred, i) => (
                      <div key={i} className="flex justify-between items-center text-xs bg-slate-900/60 px-3 py-2 rounded-lg border border-slate-800">
                        <span className="text-slate-300">
                          {lang === 'ta' ? (pred.crop_tamil || pred.crop) : pred.crop} — {pred.possible_disease}
                        </span>
                        <span className="font-mono text-emerald-400 font-semibold">
                          {Math.round(pred.confidence * 100)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <p className="text-slate-300">{result.disclaimer}</p>
          )}

          <div className="p-4 rounded-xl bg-amber-950/40 border border-amber-800/60 text-amber-300 text-xs leading-relaxed flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 flex-shrink-0 text-amber-400 mt-0.5" />
            <span>{result.disclaimer}</span>
          </div>
        </div>
      )}
    </div>
  );
};
