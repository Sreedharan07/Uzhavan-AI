export interface DialectInfo {
  dialect: string;
  dialect_code: string;
  confidence: number;
  all_scores?: Record<string, number>;
}

export interface IntentInfo {
  intent: string;
  confidence: number;
}

export interface AssistantResponse {
  transcription: string;
  normalized_transcription?: string;
  dialect?: DialectInfo;
  intent?: IntentInfo;
  response: string;
  response_ta?: string;
  response_en?: string;
  sources: string[];
  tts_audio_url?: string;
  tts_audio_url_ta?: string;
  tts_audio_url_en?: string;
  confidence: number;
  verification_badge?: string;
  ahi_score?: number;
  timestamp: string;
}

export interface CropPrediction {
  crop: string;
  crop_tamil: string;
  possible_disease: string;
  confidence: number;
}

export interface CropAnalysisResponse {
  predictions: CropPrediction[];
  top_prediction: CropPrediction;
  disclaimer: string;
}

export interface MultimodalCropResponse {
  crop: string;
  disease: string;
  predicted_class: string;
  visual_confidence: number;
  multimodal_confidence: number;
  fusion_weights: { visual?: number; symptom?: number };
  symptom_alignment: string;
  symptoms_identified: string[];
  transcribed_symptom_text: string;
  treatment: {
    disease_tamil?: string;
    symptoms?: string;
    treatment?: string;
    verified?: boolean;
  };
  disclaimer: string;
}

export interface DialectConditionedASRResponse {
  verbatim_transcription: string;
  normalized_transcription: string;
  dialect: string;
  dialect_confidence: number;
  latency_ms: number;
  audio_duration: number;
  conditioning_applied: boolean;
  diglossia_metadata: Record<string, any>;
}

export interface ModelStatus {
  status: string;
  asr_loaded: boolean;
  dialect_loaded: boolean;
  intent_loaded: boolean;
  crop_loaded: boolean;
  device: string;
  gpu_name: string;
  vram: string;
}

export interface Scheme {
  scheme_name_tamil: string;
  scheme_name_english: string;
  department: string;
  eligibility: string;
  eligibility_english?: string;
  benefits: string;
  benefits_english?: string;
  documents_required: string[];
  documents_required_english?: string[];
  application_procedure: string;
  application_procedure_english?: string;
  source_url: string;
  region: string;
  disclaimer?: string;
  disclaimer_english?: string;
}

const API_BASE = '/api';

export async function checkModelStatus(): Promise<ModelStatus> {
  const res = await fetch(`${API_BASE}/model-status`);
  if (!res.ok) throw new Error('Failed to fetch status');
  return res.json();
}

export async function sendVoiceQuery(audioBlob: Blob): Promise<AssistantResponse> {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'speech.wav');

  const res = await fetch(`${API_BASE}/assistant/query`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Assistant query failed');
  }

  return res.json();
}

export async function sendTextQuery(text: string): Promise<AssistantResponse> {
  const res = await fetch(`${API_BASE}/assistant/query?text=${encodeURIComponent(text)}`, {
    method: 'POST',
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Text query failed');
  }

  return res.json();
}

export async function analyzeCropImage(imageFile: File): Promise<CropAnalysisResponse> {
  const formData = new FormData();
  formData.append('image', imageFile);

  const res = await fetch(`${API_BASE}/crop/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Crop analysis failed');
  }

  return res.json();
}

export async function getSchemes(): Promise<{ schemes: Scheme[]; disclaimer: string }> {
  const res = await fetch(`${API_BASE}/schemes`);
  if (!res.ok) throw new Error('Failed to load schemes');
  return res.json();
}

export async function getHistory(): Promise<{ history: any[]; total: number }> {
  const res = await fetch(`${API_BASE}/history`);
  if (!res.ok) throw new Error('Failed to load history');
  return res.json();
}

export async function analyzeCropMultimodal(
  imageFile: File,
  symptomText?: string,
  symptomAudioBlob?: Blob
): Promise<MultimodalCropResponse> {
  const formData = new FormData();
  formData.append('image', imageFile);
  if (symptomAudioBlob) {
    formData.append('symptom_audio', symptomAudioBlob, 'symptom.wav');
  }

  const queryParams = symptomText ? `?symptom_text=${encodeURIComponent(symptomText)}` : '';
  const res = await fetch(`${API_BASE}/crop/multimodal-analyze${queryParams}`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Multimodal crop consultation failed');
  }

  return res.json();
}

export async function transcribeDialectConditioned(
  audioBlob: Blob,
  dialect?: string
): Promise<DialectConditionedASRResponse> {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'speech.wav');

  const queryParams = dialect ? `?dialect=${encodeURIComponent(dialect)}` : '';
  const res = await fetch(`${API_BASE}/voice/transcribe-dialect-conditioned${queryParams}`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Dialect-conditioned transcription failed');
  }

  return res.json();
}

export async function getResearchBenchmarkSummary(): Promise<any> {
  const res = await fetch(`${API_BASE}/research/benchmark-summary`);
  if (!res.ok) throw new Error('Failed to fetch research benchmark summary');
  return res.json();
}

export interface RecommendedCropItem {
  rank: number;
  crop_id: string;
  crop: string;
  crop_en: string;
  confidence: number;
  confidence_pct: string;
  yield_per_acre: string;
  duration_days: number;
  water_need: string;
  npk_ratio: string;
  tips_ta: string;
  season: string;
}

export interface XGBoostCropRecommendation {
  region: string;
  district: string;
  month: number;
  season: string;
  season_ta: string;
  top_crops: RecommendedCropItem[];
  agronomic_inputs?: Record<string, any>;
  performance: {
    engine: string;
    inference_time_us: number;
    inference_time_ms: number;
    speedup_factor: string;
    acceleration_technique: string;
  };
}

export interface XGBoostBenchmarkResult {
  iterations: number;
  standard_xgboost: {
    mean_us: number;
    mean_ms: number;
    p95_us: number;
    p99_us: number;
    throughput_qps: number;
  };
  fast_jit_xgboost: {
    mean_us: number;
    mean_ms: number;
    p95_us: number;
    p99_us: number;
    throughput_qps: number;
  };
  speedup_factor: number;
  speedup_summary: string;
  max_prediction_diff: number;
  numerical_equivalence: boolean;
}

export async function recommendCropXGBoost(params: {
  district?: string;
  soil_type?: string;
  water_source?: string;
  month?: number;
  n?: number;
  p?: number;
  k?: number;
  ph?: number;
  temperature?: number;
  humidity?: number;
  rainfall?: number;
}): Promise<XGBoostCropRecommendation> {
  const q = new URLSearchParams();
  if (params.district) q.set('district', params.district);
  if (params.soil_type) q.set('soil_type', params.soil_type);
  if (params.water_source) q.set('water_source', params.water_source);
  if (params.month) q.set('month', String(params.month));
  if (params.n !== undefined) q.set('n', String(params.n));
  if (params.p !== undefined) q.set('p', String(params.p));
  if (params.k !== undefined) q.set('k', String(params.k));
  if (params.ph !== undefined) q.set('ph', String(params.ph));
  if (params.temperature !== undefined) q.set('temperature', String(params.temperature));
  if (params.humidity !== undefined) q.set('humidity', String(params.humidity));
  if (params.rainfall !== undefined) q.set('rainfall', String(params.rainfall));

  const res = await fetch(`${API_BASE}/crop/recommend-xgboost?${q.toString()}`, {
    method: 'POST',
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'XGBoost crop recommendation failed');
  }

  return res.json();
}

export async function getXGBoostBenchmark(iterations = 200): Promise<XGBoostBenchmarkResult> {
  const res = await fetch(`${API_BASE}/crop/xgboost-benchmark?iterations=${iterations}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to execute XGBoost speed benchmark');
  }
  return res.json();
}

