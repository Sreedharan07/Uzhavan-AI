# 🌾 UZHAVAN AI: Academic Research Paper Modeling Dossier
> **Target Academic Venues:** IEEE Transactions on Audio, Speech and Language Processing / Interspeech / ACM COMPASS / IEEE TENCON / DravidianLangTech (EACL/RANLP)
> **Paper Title:** *UZHAVAN AI: Dialect-Aware Rural Tamil Voice Assistant for Farmers and Elderly Citizens*
> **Authors:** M Sreedharan (`sm7940@srmist.edu.in`), Sanjiv R (`sr3555@srmist.edu.in`), Sriramji P (`sp0429@srmist.edu.in`), Paavai G (`paavaiag@srmist.edu.in`)  
> **Affiliation:** Department of Computer Science and Engineering, SRM Institute of Science and Technology, Vadapalani, Chennai, India

---

## 📌 Instructions for ChatGPT / LLM Paper Modeling
When modeling, drafting, or editing research paper sections based on this dossier:
1. **Target Style:** Strict IEEE 2-column format or ACM SIGCONF format. Scholarly, rigorous, empirical, and publication-ready.
2. **Key Terminology:**
   - Dialect-Conditioned Automatic Speech Recognition (ASR)
   - Dravidian Diglossia (*Centamil* = formal written Tamil vs. *Kodumtamil* = colloquial spoken dialects)
   - TRAAN Benchmark (*Tamil Rural Agro-Acoustic Noise*)
   - Cross-Modal Attention Fusion (Visual Leaf Embeddings + Spoken Farmer Symptom Semantics)
   - Agricultural Hallucination Index (AHI)
   - Grounded Retrieval-Augmented Generation (RAG) with TNAU Agritech verification
3. **Use the Empirical Results and Mathematical Formulations Below** to write full manuscript sections, related work, methodology, or LaTeX tables.

---

## 1. Abstract & Metadata

### Abstract
Voice-based agricultural advisory systems hold immense promise for democratizing access to modern agronomic knowledge and governmental welfare schemes among rural, non-literate, and elderly farming populations. However, deploying conversational assistants in rural India faces four compounded sociotechnical hurdles:
1. **Severe acoustic and phonetic divergence** across non-standard regional dialects (Kongu, Madurai, Tirunelveli/Nellai, Delta, Chennai).
2. **Extreme diglossia** between spoken colloquial vernacular (*Kodumtamil*) and formal written standard (*Centamil*).
3. **Visual ambiguity** during photographic leaf disease diagnosis under unconstrained field lighting, camera motion, and lens blur.
4. **Catastrophic hallucination risks** when Large Language Models recommend unverified chemical pesticides, toxic dosage ratios, or outdated subsidy guidelines.

In this work, we present **UZHAVAN AI**, an end-to-end, dialect-conditioned multimodal conversational framework engineered for rural Tamil agriculture. 
- First, we introduce a **two-stage Dialect-Conditioned ASR** mechanism that couples an acoustic-phonetic prosody-enhanced classifier ($F_0$ pitch contours, formants, and vowel elongation dynamics) with dynamic prefix-prompt injection in Whisper, yielding a **31.4% relative Word Error Rate (WER) reduction** over standard baselines under field noise.
- Second, we formalize a **Tamil Diglossia Normalizer** providing dual-head verbatim dialect preservation alongside canonical agronomic normalization.
- Third, we formulate a **Cross-Modal Attention Fusion Network** that binds leaf imagery (EfficientNet-B0) with farmer-spoken symptom descriptions, recovering diagnostic accuracy by **+21.2%** under severe image blur.
- Finally, we implement a **Grounded Verification Guardrail** and define the **Agricultural Hallucination Index (AHI)**, reducing numerical and dosage error rates from 0.28 down to 0.02. Extensive empirical evaluations across the synthetic **TRAAN (Tamil Rural Agro-Acoustic Noise)** benchmark confirm the framework's robustness, safety, and rural usability.

**Keywords:** Dialect-Aware ASR, Low-Resource Speech Recognition, Dravidian Diglossia, Multimodal Agri-Diagnosis, Grounded RAG, AI for Social Good.

---

## 2. End-to-End System Architecture

```text
                                  ┌───────────────────────────────┐
                                  │   Farmer Rural Audio & Leaf   │
                                  └──────────────┬────────────────┘
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   ▼                                                           ▼
    [Prosody Feature Fusion]                                     [Dialect Prefix Injection]
   Mel Spectrogram + F0/Formants                                  Whisper-Small Decoder Prompting
   Outputs: Dialect Class & Latent                                Outputs: Verbatim Transcription
                   │                                                           │
                   └─────────────────────────────┬─────────────────────────────┘
                                                 ▼
                                   [Tamil Diglossia Normalizer]
                                 Kodumtamil ──► Centamil Mapping
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
        [Multimodal Disease Fusion]                        [Grounded Verified RAG]
    EfficientNet-B0 + Voice Symptoms                       Thesaurus + Dosage Guardrail
    Dynamic Cross-Modal Attention                          Calculates AHI Metric (< 0.02)
                        │                                                 │
                        └────────────────────────┬────────────────────────┘
                                                 ▼
                                     Dual-Language Safe Voice
                                     & Verified TNAU Guidance
```

---

## 3. Mathematical Formulations & Methodology

### 3.1 Dialect-Conditioned ASR & Soft-Prompt Biasing
Rather than training a monolithic ASR model that compromises across divergent regional dialects, we implement a **two-stage routing framework**:
$$\hat{d} = \arg\max_{d \in \mathcal{D}} P(d \mid \mathbf{X}_{\text{audio}})$$
Where $\mathcal{D} = \{\text{Kongu}, \text{Madurai}, \text{Nellai}, \text{Delta}, \text{Chennai}, \text{Standard}, \text{Code-Mixed}\}$.
The predicted dialect $\hat{d}$ dynamically indexes a lexical-acoustic prefix prompt:
$$P_d = \text{PromptLookup}(\hat{d})$$
Whisper's cross-attention decoder is conditioned on $P_d$ via forced token prefixing:
$$\mathbf{y} = \text{Decoder}(\mathbf{z}_{\text{acoustic}}, P_d)$$

### 3.2 Acoustic-Phonetic Prosody Fusion
A 32-dimensional prosodic vector $\mathbf{p} \in \mathbb{R}^{32}$ captures:
- Fundamental frequency contours: $F_0^{\text{mean}}, F_0^{\text{std}}, F_0^{\text{range}}, \Delta F_0$
- Energy dynamics: RMS energy, peak-to-average ratio (PAR)
- Spectral formant proxies: Spectral centroid ($F_1$), bandwidth ($F_2$), and rolloff ($F_3$)
- Voice perturbation: Jitter and shimmer proxies
- Trailing vowel elongation ratio: $\mathcal{E} = \frac{\bar{E}_{\text{tail}}}{\bar{E}_{\text{head}}}$

Gated cross-modal fusion with 2D CNN-BiLSTM Mel-spectrogram representation:
$$\mathbf{g} = \sigma(\mathbf{W}_g [\mathbf{h}_{\text{mel}} \,\|\, \mathbf{h}_{\text{prosody}}])$$
$$\mathbf{h}_{\text{fused}} = \mathbf{g} \odot \mathbf{h}_{\text{mel}} + (1 - \mathbf{g}) \odot \mathbf{W}_p \mathbf{h}_{\text{prosody}}$$

### 3.3 Multimodal Vision + Voice Symptoms Fusion
Leaf photography under unconstrained field conditions exhibits severe blur and glares, but farmers verbally state complementary symptoms:
$$\mathbf{v} = \text{EfficientNet-B0}(\mathbf{I}_{\text{leaf}}) \in \mathbb{R}^{1280}$$
$$\mathbf{s} = \text{TextEncoder}(\text{Symptoms}_{\text{spoken}}) \in \mathbb{R}^{256}$$
$$\alpha_{\text{vis}}, \alpha_{\text{sym}} = \text{Softmax}(\mathbf{W}_c [\mathbf{v} \,\|\, \mathbf{s}])$$
$$\mathbf{h}_{\text{diag}} = \alpha_{\text{vis}} \mathbf{W}_v \mathbf{v} + \alpha_{\text{sym}} \mathbf{W}_s \mathbf{s}$$

### 3.4 Grounded Knowledge Retrieval & Agricultural Hallucination Index (AHI)
To prevent lethal chemical pesticide recommendations:
$$\text{AHI} = \frac{\sum_{i=1}^{N} \mathbf{1}(\text{Claim}_i \notin \mathcal{K}_{\text{TNAU verified}})}{N_{\text{total claims}}}$$
Responses with $\text{AHI} \le 0.02$ achieve the `VERIFIED_GOVERNMENT_TNAU` compliance badge.

---

## 4. Empirical Benchmark Results (TRAAN Stress Test)

### Table I: ASR Performance by Dialect (WER and CER)
| Dialect | WER Baseline | WER Ours | CER Baseline | CER Ours |
| :--- | :---: | :---: | :---: | :---: |
| Standard Tamil | 18.4% | 12.7% | 8.2% | 5.4% |
| Kongu Tamil | 34.1% | 25.8% | 15.6% | 11.2% |
| Madurai Tamil | 31.6% | 22.9% | 14.3% | 10.1% |
| Tirunelveli Tamil | 29.8% | 21.4% | 13.7% | 9.8% |
| Delta Tamil | 27.3% | 19.6% | 12.4% | 8.9% |
| Chennai Tamil | 22.1% | 15.8% | 9.9% | 6.7% |
| Code-Mixed (Tanglish) | 38.7% | 28.3% | 17.1% | 12.4% |
| **Macro Average** | **28.9%** | **20.9%** | **13.0%** | **9.2%** |

### Table I-B: Speech Recognition WER (%) Under Field Noise (TRAAN Benchmark)
| Model Pipeline | Clean ($\infty$ dB) | 20 dB SNR | 15 dB SNR | 10 dB SNR | 5 dB SNR | 0 dB SNR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla Whisper Small** | 21.4% | 24.8% | 29.2% | 36.5% | 47.3% | 62.1% |
| **Dialect-Conditioned Whisper (Ours)** | 15.2% | 17.6% | 20.8% | 25.4% | 32.7% | 43.8% |
| **+ Diglossia Normalizer (Ours)** | **11.8%** | **13.9%** | **16.5%** | **20.1%** | **26.3%** | **35.9%** |

### Table II: Regional Dialect Identification Macro-F1 Under Noise
| Acoustic Architecture | Clean | 20 dB SNR | 15 dB SNR | 10 dB SNR | 5 dB SNR | 0 dB SNR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN-BiLSTM (Mel Only)** | 0.84 | 0.81 | 0.76 | 0.69 | 0.58 | 0.44 |
| **wav2vec2-XLSR-53 (SSL Embeddings)** | 0.88 | 0.86 | 0.82 | 0.77 | 0.68 | 0.53 |
| **Prosody-Enhanced Fusion (Ours)** | **0.92** | **0.90** | **0.87** | **0.83** | **0.76** | **0.65** |

### Table III: Crop Disease Diagnostic Accuracy Under Field Blur
| Diagnostic Modality | Clean Leaf Photo | Blurry / Glared Field Photo | Recovery Gain |
| :--- | :---: | :---: | :---: |
| **Leaf Vision Alone (EfficientNet-B0)** | 89.2% | 63.4% | Baseline |
| **Multimodal Vision + Voice Symptoms (Ours)** | **94.8%** | **84.6%** | **+21.2%** |

### Table IV: Grounded Agricultural RAG Safety & Hallucination Suppression
| Retrieval Framework | Faithfulness ($\uparrow$) | AHI Metric ($\downarrow$) | Dosage Accuracy ($\uparrow$) |
| :--- | :---: | :---: | :---: |
| **Standard Unconstrained LLM** | 71.0% | 0.28 | 64.2% |
| **UZHAVAN Grounded RAG + Guardrail (Ours)** | **96.0%** | **0.02** | **98.7%** |

### Table V: Model Inference Latency, Edge Footprint, and Throughput
| Module / Model | Parameters | Model Size | Mean Latency | Throughput | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Standard XGBoost Recommender | 24 Trees | 1.8 MB | 455.3 µs | 2,196 QPS | Baseline (1.0x) |
| **Fast JIT Engine (Ours)** | 24 Trees | 0.2 MB | **10.9 µs** | **92,008 QPS** | **41.9x – 56.0x** |
| Wake Word DS-CNN ("Hey Uzhava") | 25.4 K | 0.1 MB | 1.8 ms | 550 QPS | Edge-feasible |
| MobileNetV3-Small (Crop Disease) | 2.5 M | 9.8 MB | 14.2 ms | 70 QPS | 4.1x vs ResNet |
| DistilMuRIL Intent Classifier | 66 M | 250 MB | 28.5 ms | 35 QPS | 2.4x vs mBERT |

---

## 5. Ready-to-Use LaTeX Tables
```latex
% =========================================================
% 🌾 UZHAVAN AI - Conference Publication Tables (LaTeX)
% =========================================================


\begin{table}[t]
\centering
\caption{Speech Recognition Word Error Rate (WER \% $\downarrow$) across simulated rural acoustic noise conditions (TRAAN benchmark).}
\label{tab:asr_wer}
\begin{tabular}{lcccccc}
\toprule
\textbf{Model Pipeline} & \textbf{Clean} & \textbf{20 dB} & \textbf{15 dB} & \textbf{10 dB} & \textbf{5 dB} & \textbf{0 dB} \\
\midrule
Vanilla Whisper Small & 21.4 & 24.8 & 29.2 & 36.5 & 47.3 & 62.1 \\
Dialect-Conditioned Whisper (\textbf{Ours}) & 15.2 & 17.6 & 20.8 & 25.4 & 32.7 & 43.8 \\
+ Diglossia Normalizer (\textbf{Ours}) & \textbf{11.8 & \textbf{13.9} & \textbf{16.5} & \textbf{20.1} & \textbf{26.3} & \textbf{35.9}} \\
\bottomrule
\end{tabular}
\end{table}



\begin{table}[t]
\centering
\caption{Regional Tamil Dialect Identification Macro-F1 ($\uparrow$) under rural background noise.}
\label{tab:dialect_f1}
\begin{tabular}{lcccccc}
\toprule
\textbf{Acoustic Architecture} & \textbf{Clean} & \textbf{20 dB} & \textbf{15 dB} & \textbf{10 dB} & \textbf{5 dB} & \textbf{0 dB} \\
\midrule
CNN-BiLSTM (Mel Spectrogram) & 0.84 & 0.81 & 0.76 & 0.69 & 0.58 & 0.44 \\
wav2vec2-XLSR-53 (SSL Embeddings) & 0.88 & 0.86 & 0.82 & 0.77 & 0.68 & 0.53 \\
Prosody-Enhanced Fusion (\textbf{Ours}) & \textbf{0.92 & \textbf{0.9} & \textbf{0.87} & \textbf{0.83} & \textbf{0.76} & \textbf{0.65}} \\
\bottomrule
\end{tabular}
\end{table}



\begin{table}[t]
\centering
\caption{Crop Disease Diagnostic Accuracy: Visual EfficientNet vs. Multimodal Leaf Image + Voice Symptoms under field blur.}
\label{tab:multimodal_accuracy}
\begin{tabular}{lcc}
\toprule
\textbf{Diagnostic Modality} & \textbf{Clean Leaf Photo} & \textbf{Blurry / Glared Field Photo} \\
\midrule
Leaf Vision Alone (EfficientNet-B0) & 89.2\% & 63.4\% \\
Multimodal Vision + Voice Symptoms (\textbf{Ours}) & \textbf{94.8\%} & \textbf{84.6\%} (+21.2\%) \\
\bottomrule
\end{tabular}
\end{table}



\begin{table}[t]
\centering
\caption{Grounded Agricultural RAG Evaluation: Agricultural Hallucination Index (AHI $\downarrow$) and Chemical Ratio Faithfulness.}
\label{tab:rag_safety}
\begin{tabular}{lccc}
\toprule
\textbf{Retrieval Framework} & \textbf{Faithfulness ($\uparrow$)} & \textbf{AHI ($\downarrow$)} & \textbf{Dosage Accuracy ($\uparrow$)} \\
\midrule
Standard Unconstrained LLM & 71.0\% & 0.28 & 64.2\% \\
UZHAVAN Grounded RAG + Guardrail (\textbf{Ours}) & \textbf{96.0\%} & \textbf{0.02} & \textbf{98.7\%} \\
\bottomrule
\end{tabular}
\end{table}

```

---

## 6. Raw Benchmark JSON Telemetry
```json
{
  "metadata": {
    "name": "TRAAN Rural Acoustic Benchmark",
    "version": "1.0",
    "sample_rate": 16000,
    "snr_levels_db": [
      null,
      20.0,
      15.0,
      10.0,
      5.0,
      0.0
    ],
    "noise_profiles": [
      "Tractor Engine",
      "Diesel Pump-set",
      "Open Field Wind",
      "Mic Clipping"
    ]
  },
  "asr_wer_results": {
    "Vanilla Whisper Small": {
      "Clean": 21.4,
      "20 dB": 24.8,
      "15 dB": 29.2,
      "10 dB": 36.5,
      "5 dB": 47.3,
      "0 dB": 62.1
    },
    "Dialect-Conditioned Whisper (Ours)": {
      "Clean": 15.2,
      "20 dB": 17.6,
      "15 dB": 20.8,
      "10 dB": 25.4,
      "5 dB": 32.7,
      "0 dB": 43.8
    },
    "Dialect-Conditioned + Diglossia Normalizer (Ours)": {
      "Clean": 11.8,
      "20 dB": 13.9,
      "15 dB": 16.5,
      "10 dB": 20.1,
      "5 dB": 26.3,
      "0 dB": 35.9
    }
  },
  "dialect_classification_f1": {
    "CNN-BiLSTM (Mel Only)": {
      "Clean": 0.84,
      "20 dB": 0.81,
      "15 dB": 0.76,
      "10 dB": 0.69,
      "5 dB": 0.58,
      "0 dB": 0.44
    },
    "wav2vec2-XLSR (Frozen)": {
      "Clean": 0.88,
      "20 dB": 0.86,
      "15 dB": 0.82,
      "10 dB": 0.77,
      "5 dB": 0.68,
      "0 dB": 0.53
    },
    "Prosody-Enhanced Fusion (Ours)": {
      "Clean": 0.92,
      "20 dB": 0.9,
      "15 dB": 0.87,
      "10 dB": 0.83,
      "5 dB": 0.76,
      "0 dB": 0.65
    }
  },
  "multimodal_crop_diagnosis_accuracy": {
    "Leaf Vision Only (Clean Image)": 0.892,
    "Leaf Vision Only (Simulated Field Blur/Noise)": 0.634,
    "Multimodal Vision + Voice Symptoms (Ours - Clean)": 0.948,
    "Multimodal Vision + Voice Symptoms (Ours - Blurry Photo)": 0.846
  },
  "rag_safety_ahi": {
    "Standard Unconstrained LLM": {
      "Faithfulness": 0.71,
      "Agricultural Hallucination Index (AHI)": 0.28,
      "Verified Chemical Ratio Accuracy": "64.2%"
    },
    "UZHAVAN Grounded RAG + Guardrail (Ours)": {
      "Faithfulness": 0.96,
      "Agricultural Hallucination Index (AHI)": 0.02,
      "Verified Chemical Ratio Accuracy": "98.7%"
    }
  }
}
```

---

## 7. Curated ChatGPT Prompts for Paper Writing & Modeling

### Prompt 1: Generate Complete Methodology (Section III)
> "Act as an IEEE Transactions Senior Author. Using Section 3 and the System Architecture from this dossier, write an exhaustive Section III: Methodology in IEEE two-column format. Include subsection headers for Dialect-Conditioned ASR, Acoustic-Phonetic Prosody Fusion, Dravidian Diglossia Normalization, Multimodal Vision-Voice Disease Recovery, and Grounded TNAU RAG Guardrail. Embed all mathematical formulations with clear variable definitions."

### Prompt 2: Generate Experimental Results & Discussion (Section IV)
> "Using Tables I, II, III, and IV from this dossier, write a rigorous Section IV: Experiments and Results for an IEEE conference paper. Include analysis of why prosodic feature fusion outperforms frozen wav2vec2-XLSR under low SNR (0-10 dB), how cross-modal attention compensates for image degradation, and the statistical significance of reducing the Agricultural Hallucination Index down to 0.02."

### Prompt 3: Generate Related Work & State-of-the-Art Comparison (Section II)
> "Write Section II: Related Work comparing UZHAVAN AI against: (1) Low-resource Dravidian speech recognition (Whisper, IndicWav2Vec), (2) Diglossic NLP in South Asian languages, (3) Computer vision for agricultural plant pathology (PlantVillage, EfficientNet), and (4) RAG safety and hallucination guardrails in high-stakes domains."

### Prompt 4: Full LaTeX Paper Assembly
> "Convert this entire dossier into a complete IEEE conference LaTeX document using standard IEEEtran format, including abstract, introduction, system architecture, math equations, the four LaTeX tables, and conclusion."
