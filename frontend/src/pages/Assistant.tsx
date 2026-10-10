import React, { useState, useRef } from 'react';
import { Mic, Square, Volume2, Send, RefreshCw, AlertCircle } from 'lucide-react';
import { sendVoiceQuery, sendTextQuery, type AssistantResponse } from '../services/api';
import { type Language, translations } from '../utils/translations';

interface AssistantProps {
  elderlyMode: boolean;
  lang: Language;
}

export const Assistant: React.FC<AssistantProps> = ({ elderlyMode, lang }) => {
  const t = translations[lang].assistant;

  const [isRecording, setIsRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const [textInput, setTextInput] = useState('');
  const [liveTranscript, setLiveTranscript] = useState('');
  const [response, setResponse] = useState<AssistantResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const speechRecognitionRef = useRef<any>(null);

  const startRecording = async () => {
    setError(null);
    setLiveTranscript('');
    try {
      // 1. Audio MediaRecorder with noise reduction constraints
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        stream.getTracks().forEach((track) => track.stop());
        handleAudioSubmit(audioBlob, speechRecognitionRef.current?._latestTranscript);
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);

      // 2. Browser Web Speech Recognition for high-accuracy live hearing
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = lang === 'ta' ? 'ta-IN' : 'en-US';

        recognition.onresult = (event: any) => {
          let transcript = '';
          for (let i = event.resultIndex; i < event.results.length; i++) {
            transcript += event.results[i][0].transcript;
          }
          if (transcript.trim()) {
            setLiveTranscript(transcript);
            recognition._latestTranscript = transcript;
          }
        };

        recognition.onerror = () => {};
        recognition.start();
        speechRecognitionRef.current = recognition;
      }
    } catch (err: any) {
      setError(lang === 'ta' ? 'மைக்ரோஃபோன் அனுமதி கிடைக்கவில்லை.' : 'Microphone access denied.');
    }
  };

  const stopRecording = () => {
    if (speechRecognitionRef.current) {
      try {
        speechRecognitionRef.current.stop();
      } catch (e) {}
    }

    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleAudioSubmit = async (blob: Blob, recognizedText?: string) => {
    setLoading(true);
    try {
      let res: AssistantResponse;
      if (recognizedText && recognizedText.trim().length > 2) {
        // High accuracy browser transcription available
        res = await sendTextQuery(recognizedText);
      } else {
        res = await sendVoiceQuery(blob);
      }
      setResponse(res);
      const textToSpeak = lang === 'en' ? (res.response_en || res.response) : (res.response_ta || res.response);
      const audioUrl = lang === 'en' ? res.tts_audio_url_en : (res.tts_audio_url_ta || res.tts_audio_url);
      playFemaleVoice(textToSpeak, lang, audioUrl);
    } catch (err: any) {
      setError(err.message || (lang === 'ta' ? 'கோரிக்கை அனுப்ப முடியவில்லை' : 'Failed to send query'));
    } finally {
      setLoading(false);
    }
  };

  const handleTextSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!textInput.trim()) return;

    setLoading(true);
    setError(null);
    try {
      const res = await sendTextQuery(textInput);
      setResponse(res);
      setTextInput('');
      const textToSpeak = lang === 'en' ? (res.response_en || res.response) : (res.response_ta || res.response);
      const audioUrl = lang === 'en' ? res.tts_audio_url_en : (res.tts_audio_url_ta || res.tts_audio_url);
      playFemaleVoice(textToSpeak, lang, audioUrl);
    } catch (err: any) {
      setError(err.message || (lang === 'ta' ? 'கோரிக்கை அனுப்ப முடியவில்லை' : 'Failed to send query'));
    } finally {
      setLoading(false);
    }
  };

  const playFemaleVoice = (text: string, targetLang: 'ta' | 'en', backendUrl?: string) => {
    if (backendUrl) {
      const audio = new Audio(backendUrl);
      audio.play().catch(() => playBrowserSpeech(text, targetLang));
    } else {
      playBrowserSpeech(text, targetLang);
    }
  };

  const playBrowserSpeech = (text: string, targetLang: 'ta' | 'en') => {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();

    const cleanText = text.replace(/1\.\s*🇮🇳\s*தமிழ்:|2\.\s*🇬🇧\s*English:|🌱|🧪|🏛️|⚠️/g, '').trim();
    const utterance = new SpeechSynthesisUtterance(cleanText.slice(0, 300));
    utterance.lang = targetLang === 'ta' ? 'ta-IN' : 'en-US';
    utterance.rate = elderlyMode ? 0.8 : 1.0;
    utterance.pitch = 1.2; // Pleasant higher female tone

    const voices = window.speechSynthesis.getVoices();
    const femaleVoice = voices.find((v) => {
      const name = v.name.toLowerCase();
      return (
        (targetLang === 'ta' && (v.lang.includes('ta') || name.includes('tamil'))) ||
        (targetLang === 'en' && v.lang.includes('en'))
      ) && (name.includes('female') || name.includes('zira') || name.includes('veena') || name.includes('heera') || name.includes('google'));
    });

    if (femaleVoice) {
      utterance.voice = femaleVoice;
    }

    window.speechSynthesis.speak(utterance);
  };

  return (
    <div className={`max-w-4xl mx-auto px-4 py-6 space-y-8 ${elderlyMode ? 'elderly-mode' : ''}`}>
      {/* Mic Center Box */}
      <div className="glass-card rounded-3xl p-8 text-center space-y-6 border border-emerald-500/25">
        <h2 className="text-2xl font-bold text-white">{t.title}</h2>
        <p className="text-slate-300 font-medium">{t.subtitle}</p>

        {/* Big Mic Button */}
        <div className="py-6 flex flex-col items-center justify-center gap-3">
          <button
            onClick={isRecording ? stopRecording : startRecording}
            disabled={loading}
            className={`w-36 h-36 md:w-44 md:h-44 rounded-full flex flex-col items-center justify-center transition-all duration-300 transform active:scale-95 shadow-xl ${
              isRecording
                ? 'bg-red-600 text-white mic-recording scale-105 border-4 border-red-400'
                : 'bg-emerald-600 hover:bg-emerald-500 text-white border-4 border-emerald-400/40 hover:scale-105 shadow-emerald-950/50'
            }`}
          >
            {isRecording ? (
              <>
                <Square className="w-12 h-12 md:w-16 md:h-16 mb-2" />
                <span className="font-bold text-sm md:text-base">{t.stopSpeak}</span>
              </>
            ) : (
              <>
                <Mic className="w-12 h-12 md:w-16 md:h-16 mb-2" />
                <span className="font-bold text-sm md:text-base">{t.startSpeak}</span>
              </>
            )}
          </button>

          {isRecording && liveTranscript && (
            <div className="mt-2 p-3.5 bg-emerald-950/90 rounded-xl border border-emerald-700/60 text-emerald-300 text-sm max-w-lg shadow-xs">
              <span className="font-semibold text-xs text-emerald-400 block mb-1">
                🎙️ {lang === 'ta' ? 'நேரடி குரல் அறிதல் (Live Hearing):' : 'Live Speech Recognition:'}
              </span>
              "{liveTranscript}"
            </div>
          )}
        </div>

        {loading && (
          <div className="flex items-center justify-center gap-3 text-emerald-400 font-semibold text-lg">
            <RefreshCw className="w-6 h-6 animate-spin text-emerald-400" />
            <span>{t.processing} ({lang === 'ta' ? '5-10 வினாடிகளில் பதில் தயார்' : 'Ready in 5-10 seconds'})</span>
          </div>
        )}

        {error && (
          <div className="p-4 rounded-xl bg-red-950/80 border border-red-700/80 text-red-200 font-medium flex items-center justify-center gap-2">
            <AlertCircle className="w-5 h-5 flex-shrink-0 text-red-400" />
            <span>{error}</span>
          </div>
        )}

        {/* Text Input Fallback */}
        <form onSubmit={handleTextSubmit} className="flex gap-2 max-w-xl mx-auto pt-4">
          <input
            type="text"
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            placeholder={t.placeholder}
            className="flex-1 bg-emerald-950/50 border border-emerald-800/60 rounded-2xl px-5 py-3 text-slate-100 placeholder-slate-400 focus:outline-none focus:border-emerald-500 focus:bg-emerald-950/80 text-sm md:text-base shadow-xs transition-colors"
          />
          <button
            type="submit"
            disabled={loading || !textInput.trim()}
            className="px-6 py-3 rounded-2xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white font-bold flex items-center gap-2 shadow-xs transition-all"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
      </div>

      {/* Response Box */}
      {response && (
        <div className="glass-card rounded-3xl p-6 md:p-8 space-y-6 border border-emerald-500/25">
          {/* Metadata Badges */}
          <div className="flex flex-wrap gap-3 items-center justify-between border-b border-slate-700/60 pb-4">
            <div className="flex flex-wrap gap-2 items-center">
              {response.dialect && (
                <span className="px-3.5 py-1.5 rounded-full bg-emerald-950 text-emerald-300 border border-emerald-700/70 text-xs font-semibold">
                  🗣️ Dialect: {response.dialect.dialect} ({Math.round(response.dialect.confidence * 100)}%)
                </span>
              )}
              {response.intent && (
                <span className="px-3.5 py-1.5 rounded-full bg-teal-950 text-teal-300 border border-teal-700/70 text-xs font-semibold">
                  🎯 Intent: {response.intent.intent}
                </span>
              )}
              {response.verification_badge && (
                <span className="px-3.5 py-1.5 rounded-full bg-blue-950/80 text-blue-300 border border-blue-600/70 text-xs font-bold flex items-center gap-1.5">
                  🛡️ {response.verification_badge === 'VERIFIED_GOVERNMENT_TNAU' ? 'TNAU Verified Grounded' : response.verification_badge}
                  {response.ahi_score !== undefined && (
                    <span className="opacity-80 text-[10px]">(AHI: {response.ahi_score})</span>
                  )}
                </span>
              )}
            </div>

            {/* Pleasant Female Audio Controls */}
            <div className="flex gap-2">
              <button
                onClick={() => playFemaleVoice(response.response_ta || response.response, 'ta', response.tts_audio_url_ta)}
                className="px-3.5 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs flex items-center gap-1.5 shadow-sm"
              >
                <Volume2 className="w-4 h-4" />
                <span>🔊 தமிழ் பெண் குரல்</span>
              </button>

              <button
                onClick={() => playFemaleVoice(response.response_en || response.response, 'en', response.tts_audio_url_en)}
                className="px-3.5 py-2 rounded-xl bg-teal-600 hover:bg-teal-500 text-white font-bold text-xs flex items-center gap-1.5 shadow-sm"
              >
                <Volume2 className="w-4 h-4" />
                <span>🔊 English Female Voice</span>
              </button>
            </div>
          </div>

          {/* User Query & Diglossia Normalization Comparison */}
          <div className="space-y-2 bg-slate-900/50 p-4 rounded-2xl border border-emerald-900/50">
            <div>
              <p className="text-xs text-slate-400 uppercase tracking-wider font-bold">
                {t.youSaid} {lang === 'ta' ? '(பேச்சுத் தமிழ்)' : '(Spoken Verbatim)'}:
              </p>
              <p className="text-lg md:text-xl font-semibold text-emerald-300 mt-0.5">"{response.transcription}"</p>
            </div>

            {response.normalized_transcription && response.normalized_transcription !== response.transcription && (
              <div className="pt-2 border-t border-slate-800 text-xs text-slate-300">
                <span className="text-emerald-400 font-bold uppercase">
                  {lang === 'ta' ? 'செந்தமிழ் சீரமைப்பு (Diglossia Normalized): ' : 'Standard Tamil (Diglossia Normalized): '}
                </span>
                <span className="italic text-slate-200">"{response.normalized_transcription}"</span>
              </div>
            )}
          </div>

          {/* Dual Language AI Response Output */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Tamil Response Card */}
            <div className="bg-emerald-950/60 rounded-2xl p-5 border border-emerald-800/60 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider">1. 🇮🇳 தமிழ் பதில்</span>
                <span className="text-xs text-emerald-300 bg-emerald-950 px-2 py-0.5 rounded border border-emerald-800 font-semibold">Tamil Response</span>
              </div>
              <p className="text-base md:text-lg text-slate-100 leading-relaxed font-medium whitespace-pre-line">
                {response.response_ta || response.response}
              </p>
            </div>

            {/* English Response Card */}
            <div className="bg-teal-950/60 rounded-2xl p-5 border border-teal-800/60 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-teal-400 uppercase tracking-wider">2. 🇬🇧 English Response</span>
                <span className="text-xs text-teal-300 bg-teal-950 px-2 py-0.5 rounded border border-teal-800 font-semibold">English Translation</span>
              </div>
              <p className="text-base md:text-lg text-slate-100 leading-relaxed font-medium whitespace-pre-line">
                {response.response_en || response.response}
              </p>
            </div>
          </div>

          {response.sources && response.sources.length > 0 && (
            <div className="pt-3 border-t border-slate-700/60 text-xs text-slate-400 flex items-center gap-2">
              <span>{t.sources}</span>
              <span className="text-slate-200 font-medium">{response.sources.join(', ')}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
