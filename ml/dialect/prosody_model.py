"""
🌾 UZHAVAN AI - Prosody-Enhanced Dialect Classifier
Hybrid Neural Architecture fusing 2D Mel-Spectrogram spatio-temporal features
with 1D Acoustic-Phonetic Prosody vectors via Gated Cross-Attention.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple


class ProsodyEnhancedDialectClassifier(nn.Module):
    """
    Dual-Branch Dialect Classifier.
    Branch 1: CNN-BiLSTM for Mel-spectrogram spectral dynamics.
    Branch 2: MLP for acoustic-phonetic prosody (F0, formants, elongation, jitter).
    Fusion: Gated Bilateral Fusion Layer.
    """

    def __init__(
        self,
        n_mels: int = 80,
        cnn_channels: Tuple[int, ...] = (32, 64, 128),
        lstm_hidden: int = 128,
        prosody_dim: int = 32,
        prosody_hidden: int = 64,
        num_classes: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()

        # --- Branch 1: CNN + BiLSTM (Mel-Spectrogram) ---
        cnn_layers = []
        in_c = 1
        for out_c in cnn_channels:
            cnn_layers.extend([
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Dropout2d(p=dropout * 0.5),
            ])
            in_c = out_c

        self.cnn = nn.Sequential(*cnn_layers)

        # Mel frequency dimension reduced by 3 maxpools (80 -> 40 -> 20 -> 10)
        downsampled_mel_dim = n_mels // (2 ** len(cnn_channels))
        rnn_in_size = cnn_channels[-1] * downsampled_mel_dim

        self.bilstm = nn.LSTM(
            input_size=rnn_in_size,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.mel_fc = nn.Linear(lstm_hidden * 2, 128)

        # --- Branch 2: Prosody Feature MLP ---
        self.prosody_mlp = nn.Sequential(
            nn.Linear(prosody_dim, prosody_hidden),
            nn.BatchNorm1d(prosody_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(prosody_hidden, 128),
            nn.ReLU(inplace=True),
        )

        # --- Gated Cross-Feature Fusion ---
        # g = sigmoid(W_g * [h_mel, h_prosody])
        self.gate_fc = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.Sigmoid(),
        )

        # --- Classification Head ---
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        mel: torch.Tensor,
        prosody: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            mel: [Batch, 1, N_mels, Time] Mel-spectrogram tensor
            prosody: Optional [Batch, Prosody_dim] prosody vector.
                     If None, zeros are substituted (backward-compatible).
        """
        b, c, m, t = mel.shape

        # Mel branch
        x_cnn = self.cnn(mel)  # [B, C_last, M_down, T_down]
        b_c, c_out, m_down, t_down = x_cnn.shape
        x_reshaped = x_cnn.permute(0, 3, 1, 2).contiguous().view(b_c, t_down, -1)
        lstm_out, _ = self.bilstm(x_reshaped)  # [B, T_down, lstm_hidden*2]
        h_mel = self.mel_fc(lstm_out[:, -1, :])  # [B, 128]

        # Prosody branch
        if prosody is None:
            prosody = torch.zeros((b, 32), device=mel.device, dtype=mel.dtype)
        
        # Ensure prosody matches batch dimension
        if prosody.shape[0] != b:
            prosody = prosody.repeat(b, 1)

        h_prosody = self.prosody_mlp(prosody)  # [B, 128]

        # Gated fusion
        concat = torch.cat([h_mel, h_prosody], dim=-1)
        gate = self.gate_fc(concat)  # [B, 128]
        h_fused = gate * h_mel + (1.0 - gate) * h_prosody

        # Logits
        return self.classifier(h_fused)
