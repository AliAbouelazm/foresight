"""
Trajectory prediction models: TemporalCNN and TransformerPredictor.
Both accept (B, 20, 4) input and produce (B, 20, 2) coordinate predictions.
"""
import math

import torch
import torch.nn as nn


class TemporalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 40),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 20, 4) -> permute to (B, 4, 20) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        # global average pooling over time dimension
        x = x.mean(dim=2)          # (B, 64)
        x = self.head(x)           # (B, 40)
        return x.view(x.size(0), 20, 2)


class TransformerPredictor(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 3,
                 dim_feedforward: int = 256, dropout: float = 0.1, max_len: int = 20):
        super().__init__()
        self.input_proj = nn.Linear(4, d_model)

        # learned positional encoding
        self.pos_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 20, 4)
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = self.input_proj(x) + self.pos_embedding(positions)   # (B, T, d_model)

        # causal mask: each position attends only to itself and earlier positions
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        x = self.transformer(x, mask=mask, is_causal=True)        # (B, T, d_model)
        return self.output_proj(x)                                 # (B, T, 2)


def build_model(name: str) -> nn.Module:
    if name == "cnn":
        return TemporalCNN()
    if name == "transformer":
        return TransformerPredictor()
    raise ValueError(f"Unknown model name: {name!r}. Choose 'cnn' or 'transformer'.")
