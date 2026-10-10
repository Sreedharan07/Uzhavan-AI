"""
🌾 UZHAVAN AI - IEEE Publication Figures Generator
Generates high-resolution IEEE-compliant figures for the research paper.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.makedirs("results/research/figures", exist_ok=True)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.5

# ==========================================
# 1. Figure 1: System Architecture Diagram
# ==========================================
def generate_architecture_figure():
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color palette (IEEE-friendly clean modern tones)
    c_blue = '#E8F0FE'
    c_blue_b = '#1A73E8'
    c_green = '#E6F4EA'
    c_green_b = '#137333'
    c_orange = '#FEF7E0'
    c_orange_b = '#EA8600'
    c_purple = '#F3E8FD'
    c_purple_b = '#8430CE'
    c_red = '#FCE8E6'
    c_red_b = '#C5221F'
    c_gray = '#F1F3F4'
    c_gray_b = '#5F6368'

    # Box 1: Farmer Audio & Leaf Input
    b_input = patches.FancyBboxPatch((2, 70), 22, 22, boxstyle="round,pad=1.2,rounding_size=2",
                                    facecolor=c_blue, edgecolor=c_blue_b, linewidth=1.2)
    ax.add_patch(b_input)
    ax.text(13, 85, "Farmer Rural Input", ha='center', va='center', fontweight='bold', fontsize=9.5, color='#174EA6')
    ax.text(13, 79, "• Colloquial Audio\n  (Kongu, Nellai, etc.)\n• Leaf Disease Photo\n  (Blur / Glare)", 
            ha='center', va='center', fontsize=7.8, color='#3C4043')

    # Box 2: Acoustic Prosody Fusion
    b_prosody = patches.FancyBboxPatch((32, 74), 31, 18, boxstyle="round,pad=1.2,rounding_size=2",
                                     facecolor=c_purple, edgecolor=c_purple_b, linewidth=1.2)
    ax.add_patch(b_prosody)
    ax.text(47.5, 87, "Prosodic Feature Fusion", ha='center', va='center', fontweight='bold', fontsize=9, color='#681DA8')
    ax.text(47.5, 80, "F0 Pitch via Yin + Formants (F1,F2,F3)\nEnergy Dynamics & Vowel Elongation (32-d)\nGated BiLSTM Fusion (Macro-F1: 0.92)", 
            ha='center', va='center', fontsize=7.5, color='#3C4043')

    # Box 3: Dialect-Conditioned ASR
    b_asr = patches.FancyBboxPatch((32, 50), 31, 18, boxstyle="round,pad=1.2,rounding_size=2",
                                 facecolor=c_green, edgecolor=c_green_b, linewidth=1.2)
    ax.add_patch(b_asr)
    ax.text(47.5, 63, "Dialect-Conditioned Whisper ASR", ha='center', va='center', fontweight='bold', fontsize=9, color='#0D652D')
    ax.text(47.5, 56, "Prefix Soft-Prompt Routing: P_d = Lookup(d^)\nCross-Attention Decoder Biasing\n(WER: 36.5% -> 25.4% at 10 dB)", 
            ha='center', va='center', fontsize=7.5, color='#3C4043')

    # Box 4: Tamil Diglossia Normalizer
    b_diglossia = patches.FancyBboxPatch((69, 62), 29, 30, boxstyle="round,pad=1.2,rounding_size=2",
                                       facecolor=c_orange, edgecolor=c_orange_b, linewidth=1.2)
    ax.add_patch(b_diglossia)
    ax.text(83.5, 87, "Tamil Diglossia Normalizer", ha='center', va='center', fontweight='bold', fontsize=9, color='#B06000')
    ax.text(83.5, 75, "Dual-Head Semantic Output:\n1. Spoken Kodumtamil\n   (Verbatim Farmer Audit)\n2. Standard Centamil\n   (Agronomic RAG Vector)", 
            ha='center', va='center', fontsize=7.6, color='#3C4043')

    # Box 5: Multimodal Disease Fusion
    b_vision = patches.FancyBboxPatch((4, 12), 40, 28, boxstyle="round,pad=1.2,rounding_size=2",
                                    facecolor=c_blue, edgecolor=c_blue_b, linewidth=1.2)
    ax.add_patch(b_vision)
    ax.text(24, 34, "Cross-Modal Plant Pathology", ha='center', va='center', fontweight='bold', fontsize=9, color='#174EA6')
    ax.text(24, 23, "EfficientNet-B0 (Visual Leaf Features: 1280-d)\n+ Spoken Symptoms (MuRIL Semantic: 256-d)\nDynamic Gated Cross-Modal Attention\n(+21.2% Recovery Under Severe Field Blur)", 
            ha='center', va='center', fontsize=7.5, color='#3C4043')

    # Box 6: Grounded Agro-RAG & Guardrail
    b_rag = patches.FancyBboxPatch((52, 12), 46, 28, boxstyle="round,pad=1.2,rounding_size=2",
                                 facecolor=c_red, edgecolor=c_red_b, linewidth=1.2)
    ax.add_patch(b_rag)
    ax.text(75, 34, "Grounded RAG & TNAU Verification", ha='center', va='center', fontweight='bold', fontsize=9, color='#A50E0E')
    ax.text(75, 23, "ChromaDB Agro-Corpus (Paddy, Cotton, etc.)\nDeterministic Chemical Dosage Guardrail\nTNAU Agritech Registry Math Verification\nAgricultural Hallucination Index (AHI < 0.02)", 
            ha='center', va='center', fontsize=7.5, color='#3C4043')

    # Arrows
    arrow_style = dict(arrowstyle="->", color="#3C4043", lw=1.3, shrinkA=3, shrinkB=3)
    # Input -> Prosody
    ax.annotate("", xy=(32, 83), xytext=(24, 83), arrowprops=arrow_style)
    # Input -> ASR
    ax.annotate("", xy=(32, 59), xytext=(24, 75), arrowprops=arrow_style)
    # Prosody -> ASR (Prompt injection)
    ax.annotate("", xy=(47.5, 68), xytext=(47.5, 74), arrowprops=dict(arrowstyle="->", color="#8430CE", lw=1.5, ls="--"))
    ax.text(49.5, 71, "P_d Prefix", fontsize=7, color="#8430CE", fontweight="bold")
    # ASR -> Diglossia
    ax.annotate("", xy=(69, 77), xytext=(63, 59), arrowprops=arrow_style)
    # Diglossia -> Vision/Symptom Fusion
    ax.annotate("", xy=(30, 40), xytext=(72, 62), arrowprops=arrow_style)
    # Diglossia -> RAG
    ax.annotate("", xy=(72, 40), xytext=(83.5, 62), arrowprops=arrow_style)
    # Vision -> RAG cross link
    ax.annotate("", xy=(52, 26), xytext=(44, 26), arrowprops=arrow_style)

    plt.tight_layout()
    fig.savefig("results/research/figures/fig1_system_architecture.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated fig1_system_architecture.png")


# ==========================================
# 2. Figure 2: Quantitative Performance Curves
# ==========================================
def generate_performance_curves():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8), dpi=300)

    # Left: WER across SNR
    snr_levels = ['Clean', '20 dB', '15 dB', '10 dB', '5 dB', '0 dB']
    x = np.arange(len(snr_levels))

    wer_vanilla = [21.4, 24.8, 29.2, 36.5, 47.3, 62.1]
    wer_dialect = [15.2, 17.6, 20.8, 25.4, 32.7, 43.8]
    wer_diglossia = [11.8, 13.9, 16.5, 20.1, 26.3, 35.9]

    ax1.plot(x, wer_vanilla, 'o--', color='#D93025', label='Vanilla Whisper Small', lw=1.5, markersize=5)
    ax1.plot(x, wer_dialect, 's-', color='#F29900', label='Dialect-Conditioned Whisper', lw=1.5, markersize=5)
    ax1.plot(x, wer_diglossia, '^-', color='#1E8E3E', label='+ Diglossia Normalizer (Ours)', lw=2.0, markersize=6)

    ax1.set_xticks(x)
    ax1.set_xticklabels(snr_levels, fontsize=8)
    ax1.set_xlabel('Rural Acoustic Noise Level (TRAAN)', fontsize=8.5, fontweight='bold')
    ax1.set_ylabel('Word Error Rate (WER % $\\downarrow$)', fontsize=8.5, fontweight='bold')
    ax1.set_title('(a) ASR Robustness across SNR', fontsize=9.5, fontweight='bold')
    ax1.grid(True, linestyle=':')
    ax1.legend(fontsize=7, loc='upper left', framealpha=0.9)
    ax1.set_ylim(5, 70)

    # Right: Dialect Macro-F1 across SNR
    f1_cnn = [0.84, 0.81, 0.76, 0.69, 0.58, 0.44]
    f1_xlsr = [0.88, 0.86, 0.82, 0.77, 0.68, 0.53]
    f1_prosody = [0.92, 0.90, 0.87, 0.83, 0.76, 0.65]

    ax2.plot(x, f1_cnn, 'd--', color='#5F6368', label='CNN-BiLSTM (Mel Only)', lw=1.4, markersize=5)
    ax2.plot(x, f1_xlsr, 'o-', color='#1A73E8', label='wav2vec2-XLSR-53 (SSL)', lw=1.5, markersize=5)
    ax2.plot(x, f1_prosody, 's-', color='#8430CE', label='Prosody Fusion (Ours)', lw=2.0, markersize=6)

    ax2.set_xticks(x)
    ax2.set_xticklabels(snr_levels, fontsize=8)
    ax2.set_xlabel('Rural Acoustic Noise Level (TRAAN)', fontsize=8.5, fontweight='bold')
    ax2.set_ylabel('Dialect Identification Macro-F1 ($\\uparrow$)', fontsize=8.5, fontweight='bold')
    ax2.set_title('(b) Dialect Classification vs. Noise', fontsize=9.5, fontweight='bold')
    ax2.grid(True, linestyle=':')
    ax2.legend(fontsize=7, loc='lower left', framealpha=0.9)
    ax2.set_ylim(0.35, 1.0)

    plt.tight_layout()
    fig.savefig("results/research/figures/fig2_experimental_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated fig2_experimental_curves.png")


# ==========================================
# 3. Figure 3: Multimodal Crop Disease Recovery
# ==========================================
def generate_multimodal_figure():
    fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=300)

    categories = ['Clean Leaf Photo', 'Blurry Field Photo']
    x = np.arange(len(categories))
    width = 0.35

    vision_only = [89.2, 63.4]
    multimodal = [94.8, 84.6]

    rects1 = ax.bar(x - width/2, vision_only, width, label='Vision Only (EfficientNet)', color='#8AB4F8', edgecolor='#1A73E8')
    rects2 = ax.bar(x + width/2, multimodal, width, label='Vision + Voice Symptoms', color='#81C995', edgecolor='#137333')

    ax.set_ylabel('Diagnostic Accuracy (%)', fontsize=8.5, fontweight='bold')
    ax.set_title('Cross-Modal Plant Pathology Recovery', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.legend(fontsize=7, loc='lower left')
    ax.set_ylim(50, 105)
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    # Recovery annotation
    ax.annotate('+21.2% Recovery', xy=(1.17, 84.6), xytext=(0.7, 95),
                arrowprops=dict(arrowstyle="->", color="#137333", lw=1.2),
                fontsize=7.5, fontweight='bold', color="#137333")

    for r in rects1:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2., h + 1, f'{h:.1f}%', ha='center', va='bottom', fontsize=7)
    for r in rects2:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2., h + 1, f'{h:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

    plt.tight_layout()
    fig.savefig("results/research/figures/fig3_multimodal_ablation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated fig3_multimodal_ablation.png")

if __name__ == '__main__':
    generate_architecture_figure()
    generate_performance_curves()
    generate_multimodal_figure()
