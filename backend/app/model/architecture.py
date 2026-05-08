"""
Trajectory prediction models: TemporalCNN and TransformerPredictor.
Both accept (B, INPUT_LEN, 4) and produce (B, TARGET_LEN, 2).
"""
import torch
import torch.nn as nn

INPUT_LEN = 50
TARGET_LEN = 50


class TemporalCNN(nn.Module):
    def __init__(self, input_len: int = INPUT_LEN, target_len: int = TARGET_LEN):
        super().__init__()
        self.target_len = target_len
        self.conv_layers = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, target_len * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)          # (B, 4, T)
        x = self.conv_layers(x)          # (B, 128, T)
        x = x.mean(dim=2)               # (B, 128) global avg pool
        x = self.head(x)                 # (B, target_len*2)
        return x.view(x.size(0), self.target_len, 2)


class TransformerPredictor(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        input_len: int = INPUT_LEN,
        target_len: int = TARGET_LEN,
    ):
        super().__init__()
        self.target_len = target_len
        self.input_proj = nn.Linear(4, d_model)
        self.pos_embedding = nn.Embedding(input_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # project from input_len sequence to target_len predictions
        self.output_proj = nn.Linear(d_model, 2)
        # learned query tokens for future steps if target_len != input_len
        if target_len != input_len:
            self.future_proj = nn.Linear(d_model * input_len, target_len * 2)
            self._use_future_proj = True
        else:
            self._use_future_proj = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = self.input_proj(x) + self.pos_embedding(positions)  # (B, T, d_model)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        x = self.transformer(x, mask=mask, is_causal=True)       # (B, T, d_model)

        if self._use_future_proj:
            x_flat = x.reshape(B, -1)
            out = self.future_proj(x_flat)
            return out.view(B, self.target_len, 2)

        return self.output_proj(x)  # (B, T, 2) when target_len == input_len


def build_model(name: str) -> nn.Module:
    if name == "cnn":
        return TemporalCNN()
    if name == "transformer":
        return TransformerPredictor()
    raise ValueError(f"Unknown model name: {name!r}. Choose 'cnn' or 'transformer'.")
