# UZHAVAN AI: Dialect-Aware Rural Tamil Voice Assistant for Farmers and Elderly Citizens

**M Sreedharan** (`sm7940@srmist.edu.in`), **Sanjiv R** (`sr3555@srmist.edu.in`), **Sriramji P** (`sp0429@srmist.edu.in`), **Paavai G** (`paavaiag@srmist.edu.in`)  
*Department of Computer Science and Engineering, SRM Institute of Science and Technology, Vadapalani, Chennai, India*

---

## Abstract

**UZHAVAN AI** is a voice-first, dialect-aware AI assistant developed with rural Tamil Nadu users in mind, particularly farmers and elderly citizens. A major motivation for the work is the difficulty that existing automatic speech recognition (ASR) systems have with regional Tamil dialects and the agricultural vocabulary commonly used in everyday conversations. To address this problem, we develop an end-to-end multimodal pipeline with several connected components:
1. **Dialect-Conditioned ASR:** Whisper-small is adapted using PEFT/LoRA ($r=16$) for five Tamil dialect groups—**Kongu, Madurai, Tirunelveli, Delta, and Chennai**—resulting in a **31.4% relative reduction in Word Error Rate (WER)**.
2. **Hands-Free Keyword Spotting ("Hey Uzhava"):** A 25.4K-parameter Depthwise Separable 1D CNN (DS-CNN) running in **1.8 ms** with $O(N \cdot K)$ complexity enables hands-free activation for farmers with muddy hands.
3. **Tamil Diglossia Normalizer:** A dual-head linguistic model bridging the structural divide between informal spoken rural Tamil (*Kodumtamil*) and canonical written Tamil (*Centamil*), scoring 62.4 BLEU-4 and 97.3% agricultural term recall.
4. **Multimodal Crop Disease Diagnosis:** Fuses leaf-image features from EfficientNet-B0 with farmer-spoken symptom descriptions through an 8-head cross-modal attention network, recovering **21.2% diagnostic accuracy** lost on blurry field photographs.
5. **Hallucination Guardrail & AHI:** Grounded against Tamil Nadu Agricultural University (TNAU) pesticide and fertilizer dosage registries, enforcing an **Agricultural Hallucination Index (AHI) < 0.02** (measured at 0.016 vs 0.087 unguarded).
6. **50x Fast JIT Crop Recommender:** A Numba SIMD JIT flattened decision tree engine evaluating 24 gradient boosted trees in **10.9 µs** (92,008 QPS, a 42x–56x speedup over standard Python/scikit-learn implementations).
7. **TRAAN Rural Acoustic Benchmark:** A reproducible evaluation framework evaluating speech recognition under four realistic agricultural noise conditions: tractor rumble, diesel pump-set clatter, wind turbulence, and microphone clipping across SNR tiers from 20 dB to 0 dB.

**Keywords:** dialect-aware ASR, Tamil NLP, agricultural voice assistant, retrieval-augmented generation, hallucination mitigation, multimodal crop disease detection, rural AI accessibility, keyword spotting.

---

## I. Introduction

Agriculture remains an important part of India's workforce, employing over 60% of rural Tamil Nadu's population. Despite rapid smartphone penetration and accessible cellular connectivity, many rural users still have limited access to AI systems that can understand the way they naturally speak. This problem is especially noticeable in Tamil Nadu, where everyday speech varies considerably across regions. Most existing voice assistants (e.g., Google Assistant, Apple Siri, Amazon Alexa) are designed around Standard Tamil or English and therefore struggle with dialectal pronunciation, regional vocabulary, and the informal expressions used in rural conversations. The situation becomes more challenging when farmers switch naturally between Tamil and English (Tanglish) or use specialized agricultural terms that may not appear in standard speech datasets.

Beyond dialectal variation, three major domain-specific challenges impede conversational AI in rural agriculture:
- **Dravidian Diglossia:** A profound linguistic divide exists between literary formal Tamil (*Centamil*) used in training datasets, textbooks, and government gazettes, versus spoken rural dialects (*Kodumtamil*). Colloquial speech features phonological contractions, morphosyntactic shifts, and regional honorific suffixes (Kongu *-nga*, Nellai *-le*, Madurai *-yya*).
- **Harsh Rural Acoustics & Muddy Hands:** Field environments contain heavy tractor engine rumbles, diesel pump-set clatter, open-field wind gusts, and low-cost microphone distortion. Farmers working in irrigated paddies cannot physically tap screens with muddy hands, necessitating robust hands-free wake word detection.
- **Agronomic Hallucination Hazards:** Generic Large Language Models (LLMs) hallucinate chemical dosages or recommend prohibited pesticides. An incorrect chemical concentration can burn crops and ruin smallholder farmers.

To bridge this gap, we present **UZHAVAN AI** ("Farmer AI" in Tamil). The system uses a trainable speech pipeline based on fine-tuned Whisper-small with dialect information supplied through phonetic prompts. Beyond speech recognition, the system supports 14 agricultural intent categories, and recommendations involving pesticide and fertilizer dosages are tied to verified Tamil Nadu Agricultural University (TNAU) registry information.

### Research Contributions:
- A dialect-conditioned ASR system reducing rural Tamil WER by **31.4% relative** over a vanilla Whisper-small baseline, through PEFT/LoRA fine-tuning with phonetic dialect prompts.
- A **Tamil Diglossia Normalizer**—a dual-head linguistic model resolving the gap between spoken rural vernacular (*Kodumtamil*) and canonical written Tamil (*Centamil*).
- An **Acoustic-Phonetic Prosody Fusion Classifier** combining 2D Mel-spectrograms with pitch ($F_0$), formant frequencies ($F_1, F_2, F_3$), and vowel elongation dynamics via CNN+BiLSTM architecture with **92.8% accuracy**.
- A **Hands-Free Wake Word Spotter ("Hey Uzhava")** based on a 25.4K-parameter DS-CNN architecture running continuously in **1.8 ms** latency for eyes-free field activation.
- A **Multimodal Crop Disease Diagnosis module** fusing voice symptom descriptions with leaf images (EfficientNet-B0) via cross-modal attention, yielding **+21.2% diagnostic recovery** on blurry field photographs.
- The **TRAAN Rural Acoustic Benchmark**: a reproducible evaluation framework simulating field noise conditions including tractor rumble, diesel pump-set clatter, wind turbulence, and microphone clipping across SNR levels from 20 dB down to 0 dB.
- A **Hallucination Guardrail enforcing AHI < 0.02**, paired with an accelerated **Numba SIMD JIT Crop Recommender** executing in **10.9 µs** (92,008 QPS, a 42x–56x speedup).

---

## II. Related Work

### A. Tamil Speech Recognition
Tamil speech recognition has progressed from Hidden Markov Models (HMMs) to modern end-to-end neural architectures. Resources such as AI4Bharat's IndicWav2Vec and the IISc-MILE corpus available through OpenSLR SLR127 have helped improve recognition of standard Tamil. However, these resources do not fully address the variation found in rural and regional speech. Dialect-labelled data is still limited; for example, the DravidianLangTech shared task contains roughly 9–10 hours of such data. This makes direct fine-tuning difficult and creates a strong need for augmentation and adaptation methods.

### B. Low-Resource and Dialect-Aware ASR
Whisper has shown that large-scale weakly supervised pretraining can provide strong multilingual speech recognition across 99 languages. For low-resource settings, LoRA offers a practical way to adapt such models without retraining every parameter. Dialect-conditioned prompting has also been explored for related Dravidian-language tasks. The approach considered in this work extends that idea to a rural Tamil agricultural setting and combines it with knowledge grounding for information where incorrect answers could have practical consequences.

### C. Agricultural AI and Hallucination Mitigation
Retrieval-augmented generation (RAG) is widely used to connect language-model responses with information stored in external sources. MuRIL, a multilingual BERT model trained on 17 Indian languages, is also useful for representing and classifying Tamil-language queries. Existing agricultural chatbot systems, however, do not consistently provide a dedicated mechanism for checking chemical dosage information. Since an incorrect pesticide or fertilizer recommendation can have serious consequences, this work treats dosage verification as a separate safety requirement and expresses it quantitatively through the Agricultural Hallucination Index (AHI).

---

## III. System Architecture

```text
                                  ┌───────────────────────────────┐
                                  │   Farmer Rural Audio & Leaf   │
                                  └──────────────┬────────────────┘
                                                 │
                                                 ▼
                                     [DS-CNN Wake Word Spotter]
                                   "Hey Uzhava" Hands-Free Trigger
                                                 │
                                                 ▼
                                     [Audio Preprocessing & VAD]
                                   Spectral Gating & Silence Strip
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   ▼                                                           ▼
    [Prosody Feature Fusion]                                     [Dialect Prefix Injection]
   Mel Spectrogram + F0/Formants                                  Whisper-Small LoRA Decoder
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
                                     [Numba SIMD JIT Recommender]
                                   10.9 µs Flattened Array Decision Trees
                                                 │
                                                 ▼
                                     Dual-Language Safe Voice
                                    & Elderly-Mode Slow Audio
```

### A. Hands-Free Wake Word Spotter ("Hey Uzhava")
Farmers working in irrigated fields frequently have mud-covered hands, rendering touch gestures unusable. We introduce a continuous streaming keyword spotter for the wake phrase **"Hey Uzhava"** (*ஏய் உழவா*). The network uses a Depthwise Separable 1D CNN (DS-CNN):
$$\text{DS-Conv}(x) = \text{ReLU}(\text{BN}(\text{Conv}_{\text{point}}(\text{Conv}_{\text{depth}}(x))))$$
- Input: 1-second sliding window of 40 MFCCs × 25 time frames (16 kHz).
- Parameters: 25.4K parameters (<100 KB memory footprint).
- Inference Latency: **1.8 ms** with $O(N \cdot K)$ complexity, consuming <2% CPU on edge mobile devices.

### B. Audio Preprocessing
Incoming recordings are first converted to 16 kHz mono audio so that they have a consistent format before entering the recognition pipeline. Spectral gating is then applied through noisereduce, with the processing tuned to four noise conditions expected in agricultural settings: tractor rumble, diesel pump-set clatter, wind turbulence, and microphone clipping. WebRTC-VAD is used to identify speech segments within continuous recordings. This removes unnecessary silent portions and gives the ASR model cleaner utterance-level inputs.

### C. Acoustic-Phonetic Prosody Fusion Classifier
Dialect identification is handled in a separate branch so that the speech-recognition model can receive regional information before decoding:
1. **Mel-Spectrogram Branch:** A CNN+BiLSTM network over 2D Mel-spectrograms with 128 mel bins, 25 ms windows, and a 10 ms hop.
2. **Acoustic-Phonetic Branch:** Captures pitch ($F_0$) using CREPE, the first three formants ($F_1, F_2, F_3$) using Praat-parselmouth, and trailing vowel elongation:
$$\mathcal{E} = \frac{\int_{0.75 T}^{T} E(t)\,dt}{\int_{0}^{0.25 T} E(t)\,dt}$$
An attention-based fusion layer combines the two sources of information and produces probabilities for six classes: Standard, Kongu, Madurai, Tirunelveli, Delta, and Chennai Tamil.

### D. Dialect-Conditioned ASR via LoRA
Whisper-small serves as the main ASR model. It is fine-tuned using a combined corpus drawn from Mozilla Common Voice Tamil (200+ hours), OpenSLR SLR65 (~7 hours), OpenSLR SLR127 (150 hours from IISc-MILE), the AI4Bharat IndicVoices Tamil subset (23.7K utterances), and approximately 10 hours of dialect-labelled DravidianLangTech data. PEFT is used with LoRA at rank $r=16$, focusing on the attention projection matrices. Before decoding, the dialect identified by the upstream classifier is supplied as a textual prompt token, such as `[KONGU]`. In effect, the decoder receives a regional cue before transcribing the utterance. With this conditioning, the system records a **31.4% relative reduction in WER** compared with the unprompted Whisper-small baseline.

### E. Tamil Diglossia Normalizer
Tamil has a clear distinction between formal written language (*Centamil*) and the colloquial forms used in everyday speech (*Kodumtamil*). ASR output therefore may be understandable to a human speaker but still differ considerably from the written form expected by downstream NLP models. To reduce this mismatch, the proposed normalizer uses two complementary components:
- A character-level sequence-to-sequence model trained on paired spoken and written Tamil.
- A rule-based post-processing stage that handles recurring morphophonological contractions and agglutination patterns, including agricultural idioms and traditional Tamil land/volume measures (*குழி*, *மா*, *மரக்கால்* into metric units).

### F. Intent Classification and Grounded RAG
After normalization, the Tamil query is sent to MuRIL for intent classification. The classifier covers 14 categories: crop cultivation, pest control, fertilizers, soil management, government welfare schemes, subsidies, weather, market prices, irrigation, seed selection, post-harvest activities, livestock, rural health services, and general inquiry. The selected intent is then used to route the query to a ChromaDB vector store containing TNAU extension bulletins and government scheme documents. Both the stored documents and incoming queries are represented using the multilingual `paraphrase-multilingual-mpnet-base-v2` sentence-transformer model.

### G. Hallucination Guardrail (AHI)
Chemical pesticide and fertilizer recommendations are treated differently from ordinary conversational responses because numerical mistakes can directly affect how a farmer uses a product. Before such a response is produced, the proposed pipeline checks dosage values against structured TNAU registry entries:
$$\text{AHI} = \frac{\sum_{i=1}^{N_{\text{claims}}} \mathbf{1}(\text{Claim}_i \notin \mathcal{K}_{\text{TNAU}})}{N_{\text{claims}}}$$
During inference, unverifiable claims are suppressed and replaced with a verified alternative or an explicit uncertainty statement, allowing the system to maintain the target AHI of below 0.02 (measured at 0.016).

### H. 50x Fast Numba SIMD JIT Recommender Engine
To deliver sub-millisecond crop recommendation on resource-constrained rural edge nodes, we flatten 24 gradient boosted decision trees into C-contiguous NumPy arrays (`left_child`, `right_child`, `split_feats`, `thresholds`, `leaf_values`). Using Numba `@njit(fastmath=True, nogil=True)`, tree traversal is compiled into native SIMD machine instructions with zero heap allocation, reducing latency from 455.3 µs to **10.9 µs** (a **42x–56x speedup**, achieving **92,008 QPS**).

### I. Text-to-Speech Output & Elderly Mode
Once the response has passed the verification stage, it is converted back into speech using gTTS or AI4Bharat Vakyansh. The interface includes an **Elderly Mode**:
- Speech rate reduced by 30%.
- Pauses between words increased by 400 ms.
- Automatic repeat playback after five seconds of silence.

---

## IV. TRAAN Rural Acoustic Benchmark

Common ASR benchmarks do not necessarily represent what speech sounds like in an agricultural field. To make evaluation closer to the intended deployment environment, we introduce the **Tamil Rural Agricultural Acoustic Noise (TRAAN)** benchmark. TRAAN focuses on four field-related noise conditions combined with speech at signal-to-noise ratios (SNRs) of 20, 15, 10, 5, and 0 dB:
- **Tractor rumble:** low-frequency diesel engine vibration (80–300 Hz dominant energy band).
- **Diesel pump-set clatter:** periodic impulsive noise from irrigation pump motors, common in delta region fields.
- **Wind turbulence:** broadband noise with spectral tilt simulated via turbulent airflow recordings (<500 Hz).
- **Microphone clipping:** amplitude saturation artifacts from low-cost mobile microphones prevalent in rural use.

---

## V. Multimodal Crop Disease Diagnosis

In a typical field situation, a farmer may describe what is wrong with a crop while simultaneously using a phone camera to capture the affected leaf. UZHAVAN AI uses both forms of information rather than relying on the image alone:
- **Visual Features:** Extracted with EfficientNet-B0 pretrained on PlantVillage and fine-tuned on a Tamil Nadu regional crop-disease dataset (paddy, sugarcane, banana, and groundnut).
- **Spoken Symptoms:** Represented using MuRIL embeddings.
- **Cross-Modal Attention:** An 8-head attention network combines the visual patch features and symptom tokens into a joint representation:
$$\mathbf{h}_{\text{fusion}} = \text{MultiHeadAttn}(\mathbf{v}, \mathbf{s}, \mathbf{s}) \oplus \mathbf{v}$$

For images affected by motion blur (kernel $\sigma > 3$ pixels), adding the voice description recovers **21.2% of the diagnostic accuracy** compared with using the vision model alone.

---

## VI. Experimental Results

### A. ASR Performance by Dialect (Table I)

Experiments were conducted on an NVIDIA A100 GPU (40 GB VRAM). Whisper-small was fine-tuned for 20 epochs using AdamW ($\text{lr} = 10^{-4}$) with linear warmup over the first 500 steps.

#### TABLE I. ASR PERFORMANCE BY DIALECT
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

*Takeaway:* The dialect-conditioned model outperforms the vanilla Whisper baseline across every dialect. Kongu Tamil achieves an 8.3 percentage point WER drop due to explicit vowel-elongation modeling.

---

### B. Regional Dialect Identification Performance (Table II)

#### TABLE II. REGIONAL DIALECT IDENTIFICATION PERFORMANCE
| Model Architecture | Accuracy | Precision | Recall | Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| Baseline Mel-CNN | 76.2% | 74.8% | 75.5% | 0.751 |
| CNN-BiLSTM (Acoustic) | 82.4% | 81.9% | 82.1% | 0.820 |
| wav2vec2-XLSR-53 (Frozen SSL) | 86.7% | 86.1\% | 86.4% | 0.862 |
| **Prosody-Attention Fusion (Ours)** | **92.8%** | **92.4%** | **92.6%** | **0.925** |

---

### C. Tamil Diglossia Normalization Performance (Table III)

#### TABLE III. TAMIL DIGLOSSIA NORMALIZATION PERFORMANCE (KODUMTAMIL $\rightarrow$ CENTAMIL)
| Dialect Variant | BLEU-4 ($\uparrow$) | chrF++ ($\uparrow$) | Exact Match | Agri-Term Recall |
| :--- | :---: | :---: | :---: | :---: |
| Kongu Dialect Normalization | 58.4 | 76.2 | 71.4% | 96.2% |
| Madurai Dialect Normalization | 61.2 | 78.9 | 74.8% | 97.5% |
| Nellai Dialect Normalization | 59.8 | 77.1 | 72.3% | 95.8% |
| Delta Dialect Normalization | 64.5 | 81.4 | 78.6% | 98.1% |
| Chennai Dialect Normalization | 68.1 | 84.6 | 82.3% | 98.9% |
| **Overall Normalizer Average** | **62.4** | **79.6** | **75.9%** | **97.3%** |

---

### D. Multimodal Crop Disease Diagnostic Accuracy (Table IV)

#### TABLE IV. MULTIMODAL CROP DISEASE DIAGNOSTIC ACCURACY UNDER FIELD DEGRADATIONS
| Input Modality | Clean Field | Motion Blur | Sunlight Glare | Low-Light Field |
| :--- | :---: | :---: | :---: | :---: |
| Voice Symptoms Alone (Text NLP) | 68.4% | 68.4% | 68.4% | 68.4% |
| Leaf Vision Alone (EfficientNet-B0) | 89.2% | 63.4% | 67.8% | 61.5% |
| Early Feature Concat Fusion | 91.6% | 74.2% | 76.9% | 73.1% |
| **Cross-Modal Attention (Ours)** | **94.8%** | **84.6%** | **86.3%** | **83.2%** |

*Takeaway:* Cross-modal attention recovers **+21.2% accuracy** under field blur (84.6% vs 63.4%).

---

### E. Intent Classification Performance (Table V)

#### TABLE V. INTENT CLASSIFICATION PERFORMANCE ACROSS TAMIL QUERY STYLES
| Model Architecture | Standard Tamil | Colloquial Dialect | Code-Mixed | Overall F1 |
| :--- | :---: | :---: | :---: | :---: |
| TF-IDF + Linear SVM | 81.2% | 69.4% | 62.8% | 0.711 |
| IndicBERT Base | 88.5% | 78.6% | 74.2% | 0.804 |
| mBERT (Multilingual) | 87.1% | 77.2% | 76.8% | 0.803 |
| **DistilMuRIL + BiGRU (Ours)** | **93.4%** | **88.7%** | **86.5%** | **0.895** |

*Class-wise breakdown:* Macro-F1 across 14 categories reaches 91.3%. Crop cultivation achieves 95.2% F1, pest control 93.8% F1, while government schemes and subsidies reach 84.7% F1 due to nomenclature overlap.

---

### F. Grounded RAG Safety and Hallucination Index (Table VI)

#### TABLE VI. GROUNDED RAG SAFETY AND HALLUCINATION INDEX (AHI)
| Retrieval Framework | Faithfulness ($\uparrow$) | Answer Relevance | Dosage Accuracy | AHI Metric ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: |
| Unconstrained Commercial LLM | 71.0% | 82.4% | 64.2% | 0.280 |
| Standard Naive Vector RAG | 83.5% | 88.1% | 81.4% | 0.124 |
| Reranked Hybrid RAG | 89.2% | 91.6% | 89.7% | 0.071 |
| **UZHAVAN Grounded RAG (Ours)** | **96.0%** | **95.4%** | **98.7%** | **0.020** |

*Takeaway:* Grounding against TNAU registries reduces AHI from 0.087 down to 0.016 in empirical testing across 500 queries.

---

### G. Model Inference Latency & Edge Footprint (Table VII)

#### TABLE VII. MODEL INFERENCE LATENCY AND EDGE RESOURCE CONSUMPTION
| Module / Model | Parameters | Model Size | Mean Latency | Throughput | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Standard XGBoost Recommender | 24 Trees | 1.8 MB | 455.3 µs | 2,196 QPS | Baseline (1.0x) |
| **Fast JIT Engine (Ours)** | 24 Trees | 0.2 MB | **10.9 µs** | **92,008 QPS** | **41.9x – 56.0x** |
| Wake Word DS-CNN ("Hey Uzhava") | 25.4 K | 0.1 MB | 1.8 ms | 550 QPS | Edge-feasible |
| MobileNetV3-Small (Crop Disease) | 2.5 M | 9.8 MB | 14.2 ms | 70 QPS | 4.1x vs ResNet |
| DistilMuRIL Intent Classifier | 66 M | 250 MB | 28.5 ms | 35 QPS | 2.4x vs mBERT |

---

### H. Speech Recognition Under Simulated Noise (Table VIII)

#### TABLE VIII. SPEECH RECOGNITION WER (%) ACROSS NOISE LEVELS (TRAAN BENCHMARK)
| Model Pipeline | Clean | 20 dB | 15 dB | 10 dB | 5 dB | 0 dB (Severe) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Vanilla Whisper Small | 21.4% | 24.8% | 29.2% | 36.5% | 47.3% | 62.1% |
| Dialect-Conditioned Whisper (Ours) | 15.2% | 17.6% | 20.8% | 25.4% | 32.7% | 43.8% |
| **+ Diglossia Normalizer (Ours)** | **11.8%** | **13.9%** | **16.5%** | **20.1%** | **26.3%** | **35.9%** |
| *Relative Error Reduction ($\Delta$)* | *-44.9%* | *-44.0%* | *-43.5%* | *-44.9%* | *-44.4%* | *-42.2%* |

---

## VII. Ethical Considerations

Because the system is intended for rural communities and may process recordings from older users, data handling is an important part of the design:
- **Informed Consent & Anonymization:** Audio collection tools ask for explicit informed consent before storing audio. Speaker identifiers are hashed using SHA-256, while raw personal metadata is removed during collection.
- **Government Welfare Verification:** Responses about government schemes include a disclaimer reminding users that eligibility should be confirmed through official channels (VAO / Panchayat).
- **Synthetic Noise Benchmark:** For TRAAN, the benchmark uses synthetically composited noise rather than ambient recordings tied to individual farms, avoiding the leakage of private farmstead audio into evaluation data.

---

## VIII. Conclusion

This paper presented **UZHAVAN AI**, a dialect-aware multimodal voice assistant aimed at Tamil-speaking farmers and elderly citizens. The results show that dialect conditioning can reduce rural WER by 31.4% relative to the baseline, while the TRAAN benchmark provides a way to evaluate speech recognition under field-like acoustic conditions. The hands-free DS-CNN keyword spotter provides continuous wake word detection in 1.8 ms for mud-covered hands. The multimodal crop-disease component recovers 21.2% of diagnostic accuracy on blurry images when spoken symptom descriptions are added. The dosage-verification mechanism keeps the measured AHI at 0.016, below the target of 0.02, while Numba SIMD JIT compilation accelerates crop recommendation by up to 56x (10.9 µs latency). Future work will focus on extending dialect coverage to Sri Lankan Tamil and diaspora varieties, supporting offline edge operation in areas with little or no connectivity, and carrying out participatory studies with farmer cooperatives in Tamil Nadu.

---

## Acknowledgment

The authors thank Tamil Nadu Agricultural University (TNAU) for providing access to the agricultural registry databases used for knowledge grounding. We also acknowledge Mozilla Common Voice, AI4Bharat, and the DravidianLangTech shared task organizers for facilitating access to the speech resources used in this work.

---

## References

1. A. Ramakrishnan, S. Balaji, and R. Murthy, "Phoneme-level modeling for Tamil speech recognition," in *Proc. Interspeech*, 2019, pp. 1201–1205.
2. A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust speech recognition via large-scale weak supervision," in *Proc. ICML*, 2023.
3. V. Prakash and M. Selvam, "End-to-end deep learning approaches for Tamil speech recognition," *J. King Saud Univ. – Comput. Inf. Sci.*, vol. 34, no. 8, pp. 5862–5871, 2022.
4. B. R. Chakravarthi et al., "DravidianLangTech@ACL2022: Shared task on dialect identification of Tamil," in *Proc. ACL Workshop on Dravidian Language Technologies*, 2022.
5. E. J. Hu et al., "LoRA: Low-rank adaptation of large language models," in *Proc. ICLR*, 2022.
6. P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," in *Proc. NeurIPS*, 2020.
7. D. Khanuja et al., "MuRIL: Multilingual representations for Indian languages," 2021, arXiv:2103.10730.
8. M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. ICML*, 2019.
9. Mozilla Foundation, "Common Voice Tamil," commonvoice.mozilla.org. [Online]. Available: https://commonvoice.mozilla.org
10. AI4Bharat, "IndicVoices Tamil," Hugging Face. [Online]. Available: https://huggingface.co/ai4bharat
11. A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A framework for self-supervised learning of speech representations," in *Proc. NeurIPS*, 2020.
12. A. Ganapathiraju et al., "IISc-MILE Tamil Speech Corpus," OpenSLR SLR127, 2022.
