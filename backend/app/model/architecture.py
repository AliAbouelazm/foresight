"""
Trajectory prediction models.
- TemporalCNN: fast baseline, single-pass regression
- Seq2SeqTrajectory: encoder-decoder transformer, autoregressively predicts
  future velocities then integrates to positions. Teacher forcing during training.
Both accept (B, INPUT_LEN, 4) and produce (B, TARGET_LEN, 2).
"""
import torch
import torch.nn as nn

INPUT_LEN = 50
TARGET_LEN = 50


class TemporalCNN(nn.Module):
    def __init__(self, target_len: int = TARGET_LEN):
        super().__init__()
        self.target_len = target_len
        self.conv = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=5, padding=2), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv1d(256, 128, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(0.1),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, target_len * 2),
        )

    def forward(self, x: torch.Tensor, target=None) -> torch.Tensor:
        h = self.conv(x.permute(0, 2, 1)).mean(dim=2)
        return self.head(h).view(x.size(0), self.target_len, 2)


class Seq2SeqTrajectory(nn.Module):
    """
    Encoder-decoder transformer for trajectory prediction.

    Encoder: processes (x, y, dx, dy) history.
    Decoder: autoregressively predicts velocity (dx, dy) at each future step,
             conditioned on encoder memory and previously predicted positions.
    Positions are recovered by integrating predicted velocities from the
    last known input position.

    During training: teacher forcing feeds ground-truth positions as decoder input.
    During inference: feeds its own predicted positions autoregressively.
    """
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_enc_layers: int = 4,
        num_dec_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        input_len: int = INPUT_LEN,
        target_len: int = TARGET_LEN,
    ):
        super().__init__()
        self.target_len = target_len

        # Encoder
        self.enc_proj = nn.Linear(4, d_model)
        self.enc_pos = nn.Embedding(input_len, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_layers=num_enc_layers,
        )

        # Decoder — input is (x, y) position of previous step
        self.dec_proj = nn.Linear(2, d_model)
        self.dec_pos = nn.Embedding(target_len, d_model)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_layers=num_dec_layers,
        )
        self.out_vel = nn.Linear(d_model, 2)  # predicts (dx, dy)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        return self.encoder(self.enc_proj(x) + self.enc_pos(pos))

    def forward(self, x: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        B = x.size(0)
        device = x.device
        memory = self._encode(x)           # (B, input_len, d_model)
        last_pos = x[:, -1, :2]            # (B, 2) — last known position

        if target is not None and self.training:
            # Teacher forcing: decoder input is [last_pos, target_0, ..., target_{T-2}]
            dec_positions = torch.cat([last_pos.unsqueeze(1), target[:, :-1]], dim=1)  # (B, T, 2)
            step_idx = torch.arange(self.target_len, device=device).unsqueeze(0).expand(B, -1)
            dec_in = self.dec_proj(dec_positions) + self.dec_pos(step_idx)

            causal = nn.Transformer.generate_square_subsequent_mask(self.target_len, device=device)
            dec_out = self.decoder(dec_in, memory, tgt_mask=causal, tgt_is_causal=True)
            velocities = self.out_vel(dec_out)                           # (B, T, 2)
            return last_pos.unsqueeze(1) + torch.cumsum(velocities, dim=1)

        # Autoregressive inference
        cur_positions = [last_pos]
        for step in range(self.target_len):
            dec_positions = torch.stack(cur_positions, dim=1)            # (B, step+1, 2)
            step_idx = torch.arange(step + 1, device=device).unsqueeze(0).expand(B, -1)
            dec_in = self.dec_proj(dec_positions) + self.dec_pos(step_idx)
            causal = nn.Transformer.generate_square_subsequent_mask(step + 1, device=device)
            dec_out = self.decoder(dec_in, memory, tgt_mask=causal, tgt_is_causal=True)
            vel = self.out_vel(dec_out[:, -1])                           # (B, 2)
            cur_positions.append(cur_positions[-1] + vel)

        return torch.stack(cur_positions[1:], dim=1)                     # (B, target_len, 2)


def build_model(name: str) -> nn.Module:
    if name == "cnn":
        return TemporalCNN()
    if name == "transformer":
        return Seq2SeqTrajectory()
    raise ValueError(f"Unknown model name: {name!r}. Choose 'cnn' or 'transformer'.")
