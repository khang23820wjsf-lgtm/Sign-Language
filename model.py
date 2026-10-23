#!/usr/bin/env python3
"""
===============================================================================
VSL CONTINUOUS CTC MODEL & SINGLE-SIGN BiLSTM BASELINE ARCHITECTURE (PyTorch)
Reproduces:
1. Primary Continuous CTC Architecture (`VSLSignCTCModel` from `scripts/train_ctc.py`):
   - Input: `[batch_size, sequence_length, 212]` (variable T, 106 Left + 106 Right)
   - Temporal Conv1D Encoder: Conv1d(212 -> 64, k=3, pad=1) + BatchNorm1d(64) + ReLU + Dropout(0.2)
   - Bidirectional LSTM Encoder: 2-layer BiLSTM(64 -> 128 per dir = 256, dropout=0.25)
   - Classification Head: Linear(256 -> 64) + ReLU + Dropout(0.2) + Linear(64 -> num_classes) + LogSoftmax
   - Includes pre/post Conv1d padding masks and `pack_padded_sequence` / `pad_packed_sequence`
     for exact batch-vs-single inference invariance (`TEST 9` in `test_train_pipeline.py`).
2. Single-Sign Classification Baseline (`VSLSingleSignBaselineModel` from `src/utils/lstmModel.ts`):
   - Input: `[batch_size, 30, 138]` (30 frames, 138 features: 106 hand + 32 Pose/NMM)
   - Conv1D(138 -> 32, k=3, pad=1) + ReLU + MaxPool1d(2) (30 -> 15 steps)
   - 1-layer BiLSTM(32 -> 32 per dir = 64, last timestep output)
   - Dropout(0.25) + Linear(64 -> 32) + ReLU + Linear(32 -> num_classes)
===============================================================================
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from feature_extractor import (
    CTC_FEATURE_DIM,
    FEATURE_VERSION,
    SINGLE_SIGN_FEATURE_DIM,
    SINGLE_SIGN_SEQUENCE_LENGTH,
)


class VSLSignCTCModel(nn.Module):
    """
    Source-compatible Continuous VSL Recognition Model (1D-CNN + 2x BiLSTM + CTC Head).
    Matches `scripts/train_ctc.py` layer definitions and parameter names (`conv1d`, `bilstm`, `fc`)
    while enforcing padding mask invariance across `Conv1d` and `BiLSTM` when `lengths` is provided.
    """

    def __init__(
        self,
        input_dim: int = CTC_FEATURE_DIM,
        hidden_dim: int = 128,
        num_classes: int = 22,
        conv_channels: int = 64,
        conv_dropout: float = 0.2,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.25,
        fc_hidden_dim: int = 64,
        fc_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.conv_channels = conv_channels
        self.lstm_layers = lstm_layers
        self.fc_hidden_dim = fc_hidden_dim

        # 1. Temporal Convolution Encoder (preserves temporal length T)
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(conv_dropout),
        )

        # 2. Bidirectional Recurrent Encoder (2-layer BiLSTM -> 256 features per time step)
        self.bilstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        # 3. Classification Head (256 -> 64 -> num_classes)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden_dim, num_classes),
        )
        # Initialize blank token (index 0) bias slightly negative to prevent early CTC blank collapse
        with torch.no_grad():
            last_linear = self.fc[-1]
            if isinstance(last_linear, nn.Linear) and last_linear.bias is not None and num_classes > 1:
                last_linear.bias.zero_()
                last_linear.bias[0] = -1.2

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass:
        Args:
            x: Tensor of shape `[batch_size, sequence_length, 212]`
            lengths: Optional 1D tensor of true sequence lengths `[batch_size]`
        Returns:
            log_probs: Tensor of shape `[batch_size, sequence_length, num_classes]`
        """
        if x.dim() != 3 or x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected 3D input tensor [B, T, {self.input_dim}], got shape {tuple(x.shape)}"
            )

        batch_size, max_t, _ = x.shape

        if lengths is not None:
            lengths_cpu = lengths.detach().to(device="cpu", dtype=torch.long)
            if torch.any(lengths_cpu <= 0):
                raise ValueError(f"All sequence lengths must be > 0, got {lengths_cpu.tolist()}")

            # Build boolean mask [B, 1, T] for Conv1d channel-first tensor [B, C, T]
            time_idx = torch.arange(max_t, device=x.device).unsqueeze(0)  # [1, T]
            mask = (time_idx < lengths.to(x.device).unsqueeze(1)).unsqueeze(1).to(x.dtype)  # [B, 1, T]

            x = x.transpose(1, 2)  # [B, 212, T]
            x = x * mask

            # Note on BatchNorm1d during training vs eval:
            # In eval mode, running stats are used per time step, so masking before & after Conv1d
            # guarantees exact mathematical invariance between single and batched inference.
            x_conv = self.conv1d(x)  # [B, 64, T]
            x_conv = x_conv * mask
            x_conv = x_conv.transpose(1, 2)  # [B, T, 64]

            packed = pack_padded_sequence(
                x_conv,
                lengths_cpu,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.bilstm(packed)
            lstm_out, _ = pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=max_t,
            )  # [B, T, 256]
        else:
            x = x.transpose(1, 2)        # [B, 212, T]
            x_conv = self.conv1d(x)      # [B, 64, T]
            x_conv = x_conv.transpose(1, 2)  # [B, T, 64]
            lstm_out, _ = self.bilstm(x_conv)  # [B, T, 256]

        logits = self.fc(lstm_out)  # [B, T, num_classes]
        log_probs = F.log_softmax(logits, dim=2)
        return log_probs

    def get_architecture_config(self) -> Dict[str, Any]:
        return {
            "architecture": "1D-CNN + 2x BiLSTM + Linear(num_classes)",
            "input_dim": self.input_dim,
            "conv_channels": self.conv_channels,
            "hidden_dim": self.hidden_dim,
            "lstm_layers": self.lstm_layers,
            "fc_hidden_dim": self.fc_hidden_dim,
            "num_classes": self.num_classes,
            "feature_version": FEATURE_VERSION,
            "sequence_format": "[T,212]",
        }


class VSLSingleSignBaselineModel(nn.Module):
    """
    Single-Sign Classification Baseline reproduced from `src/utils/lstmModel.ts`:
    - Input shape: `[batch_size, 30, 138]` (30 frames, 138 features per frame)
    - Conv1D: 138 -> conv_filters, kernel_size=3, padding=1 ('same'), BatchNorm1d, ReLU
    - MaxPool1D: pool_size=2 (30 steps -> 15 steps)
    - Bidirectional LSTM: lstm_units per direction (2*lstm_units total)
      combines final forward/backward state with temporal mean pooling over the 15 steps
    - Dropout: 0.25
    - Dense: 2*lstm_units -> dense_units, ReLU
    - Output Dense: dense_units -> num_classes (logits or softmax probabilities)
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int = SINGLE_SIGN_FEATURE_DIM,
        sequence_length: int = SINGLE_SIGN_SEQUENCE_LENGTH,
        conv_filters: int = 64,
        lstm_units: int = 64,
        dropout: float = 0.25,
        dense_units: int = 64,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"Invalid num_classes for VSLSingleSignBaselineModel: expected >= 2, got {num_classes}")

        self.num_classes = num_classes
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.conv_filters = conv_filters
        self.lstm_units = lstm_units
        self.dense_units = dense_units

        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_filters,
            kernel_size=3,
            padding=1,
        )
        self.bn1d = nn.BatchNorm1d(conv_filters)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bilstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_units * 2, dense_units)
        self.fc_out = nn.Linear(dense_units, num_classes)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape `[batch_size, 30, 138]`
        Returns:
            logits: Tensor of shape `[batch_size, num_classes]`
        """
        if x.dim() != 3 or x.size(1) != self.sequence_length or x.size(2) != self.input_dim:
            raise ValueError(
                f"Single-sign baseline expects [B, {self.sequence_length}, {self.input_dim}], got {tuple(x.shape)}"
            )

        # [B, 30, 138] -> [B, 138, 30]
        h = x.transpose(1, 2)
        h = F.relu(self.bn1d(self.conv1d(h)))  # [B, C, 30]
        h = self.pool(h)                       # [B, C, 15]
        h = h.transpose(1, 2)                  # [B, 15, C]

        # BiLSTM sequence output [B, 15, 2*H] and final hidden state (h_n: [2, B, H])
        lstm_seq, (h_n, _) = self.bilstm(h)
        h_last = torch.cat([h_n[0], h_n[1]], dim=-1)  # [B, 2*H]
        h_mean = lstm_seq.mean(dim=1)                 # [B, 2*H]
        h_bi = 0.5 * (h_last + h_mean)

        h_bi = self.dropout(h_bi)
        h_dense = F.relu(self.fc1(h_bi))
        logits = self.fc_out(h_dense)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns Softmax probabilities `[batch_size, num_classes]` matching `src/utils/lstmModel.ts`."""
        return F.softmax(self.forward_logits(x), dim=-1)

    def get_architecture_config(self) -> Dict[str, Any]:
        return {
            "architecture": f"Conv1D({self.conv_filters},k=3)+BN + MaxPool1D(2) + BiLSTM({self.lstm_units}) + Dropout(0.25) + Dense({self.dense_units}) + Softmax",
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
            "conv_filters": self.conv_filters,
            "lstm_units": self.lstm_units,
            "dense_units": self.dense_units,
            "num_classes": self.num_classes,
        }
