# Autoformer bits for Task 2 (decomp + auto-correlation).
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel: int = 25):
        super().__init__()
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError("kernel must be positive and odd")
        self.kernel = kernel

    def forward(self, x: torch.Tensor):
        radius = self.kernel // 2
        padded = F.pad(x.transpose(1, 2), (radius, radius), mode="replicate")
        trend = F.avg_pool1d(padded, kernel_size=self.kernel, stride=1).transpose(1, 2)
        return x - trend, trend


def delay_scores(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    queries = queries - queries.mean(-1, keepdim=True)
    keys = keys - keys.mean(-1, keepdim=True)
    spectrum = torch.fft.rfft(queries, dim=-1) * torch.fft.rfft(keys, dim=-1).conj()
    return torch.fft.irfft(spectrum, n=queries.shape[-1], dim=-1).mean((1, 2))


def aggregate_delays(values: torch.Tensor, delays: torch.Tensor,
                     weights: torch.Tensor) -> torch.Tensor:
    batch, heads, channels, length = values.shape
    k = delays.shape[-1]
    t = torch.arange(length, device=values.device).view(1, 1, length)
    src = (t - delays.to(values.device).unsqueeze(-1)) % length
    src = src.long().view(batch, 1, 1, k, length).expand(batch, heads, channels, k, length)
    gathered = torch.gather(
        values.unsqueeze(3).expand(batch, heads, channels, k, length), 4, src)
    w = weights.to(device=values.device, dtype=values.dtype).view(batch, 1, 1, k, 1)
    return (gathered * w).sum(dim=3)


class AutoCorrelation(nn.Module):
    def __init__(self, d_model: int, n_heads: int, factor: int = 2, dropout: float = 0.05):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.factor = factor
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        batch, q_len, d_model = queries.shape
        _, k_len, _ = keys.shape
        head_dim = d_model // self.n_heads

        q = self.query(queries).view(batch, q_len, self.n_heads, head_dim).permute(0, 2, 3, 1)
        k = self.key(keys).view(batch, k_len, self.n_heads, head_dim).permute(0, 2, 3, 1)
        v = self.value(values).view(batch, k_len, self.n_heads, head_dim).permute(0, 2, 3, 1)

        # decoder length can differ from encoder
        if k_len != q_len:
            if k_len > q_len:
                k = k[..., -q_len:]
                v = v[..., -q_len:]
            else:
                pad = q_len - k_len
                k = F.pad(k, (pad, 0))
                v = F.pad(v, (pad, 0))

        length = q.shape[-1]
        scores = delay_scores(q, k)
        max_delay = max(2, length // 2)
        band = scores[:, 1:max_delay]
        top_k = max(1, min(band.shape[-1], self.factor * int(math.ceil(math.log(max(length, 2))))))
        selected, indices = band.topk(top_k, dim=-1)
        delays = indices + 1
        weights = selected.softmax(dim=-1)
        mixed = aggregate_delays(v, delays, weights)
        mixed = mixed.permute(0, 3, 1, 2).reshape(batch, length, d_model)
        return self.dropout(self.out(mixed))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, moving_avg, factor, dropout=0.05):
        super().__init__()
        self.corr = AutoCorrelation(d_model, n_heads, factor, dropout)
        self.decomp1 = SeriesDecomposition(moving_avg)
        self.decomp2 = SeriesDecomposition(moving_avg)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.corr(self.norm1(x), self.norm1(x), self.norm1(x))
        seasonal, _ = self.decomp1(x)
        seasonal = seasonal + self.ff(self.norm2(seasonal))
        seasonal, _ = self.decomp2(seasonal)
        return seasonal


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, moving_avg, factor, c_out, dropout=0.05):
        super().__init__()
        self.self_corr = AutoCorrelation(d_model, n_heads, factor, dropout)
        self.cross_corr = AutoCorrelation(d_model, n_heads, factor, dropout)
        self.decomp1 = SeriesDecomposition(moving_avg)
        self.decomp2 = SeriesDecomposition(moving_avg)
        self.decomp3 = SeriesDecomposition(moving_avg)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.trend_proj = nn.Linear(d_model, c_out)

    def forward(self, x, cross):
        x = x + self.self_corr(self.norm1(x), self.norm1(x), self.norm1(x))
        seasonal, trend1 = self.decomp1(x)
        seasonal = seasonal + self.cross_corr(self.norm2(seasonal), cross, cross)
        seasonal, trend2 = self.decomp2(seasonal)
        seasonal = seasonal + self.ff(self.norm3(seasonal))
        seasonal, trend3 = self.decomp3(seasonal)
        return seasonal, self.trend_proj(trend1 + trend2 + trend3)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, max_len=1024, dropout=0.05):
        super().__init__()
        self.value = nn.Linear(c_in, d_model)
        # Sinusoidal PE — no trainable position table (saves parameters under the leaderboard penalty).
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div)
        pe[0, :, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.value(x) + self.pe[:, :x.shape[1]])


class Autoformer(nn.Module):
    def __init__(
        self,
        enc_in: int = 1,
        dec_in: int = 1,
        c_out: int = 1,
        seq_len: int = 336,
        label_len: int = 84,
        pred_len: int = 168,
        d_model: int = 32,
        n_heads: int = 4,
        e_layers: int = 2,
        d_layers: int = 1,
        moving_avg: int = 25,
        factor: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.c_out = c_out

        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout=dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout=dropout)
        self.decomp = SeriesDecomposition(moving_avg)
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, n_heads, moving_avg, factor, dropout)
            for _ in range(e_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, n_heads, moving_avg, factor, c_out, dropout)
            for _ in range(d_layers)
        ])
        self.projection = nn.Linear(d_model, c_out)
        # last few points -> coarse level over the horizon (lag-1 is strong here)
        self.level_head = nn.Linear(8, pred_len)

    def forward(self, x_enc, x_dec):
        # x_enc: [B, L, C], x_dec: [B, label+pred, C] -> [B, pred, c_out]
        target = x_enc[:, :, :self.c_out]
        seasonal_init, trend_init = self.decomp(target)
        mean = target.mean(dim=1, keepdim=True).repeat(1, self.pred_len, 1)
        trend_seed = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)

        enc = self.enc_embedding(x_enc)
        for layer in self.encoder:
            enc = layer(enc)
        enc = self.enc_norm(enc)

        seasonal = self.dec_embedding(x_dec)
        trend_acc = trend_seed
        for layer in self.decoder:
            seasonal, trend = layer(seasonal, enc)
            trend_acc = trend_acc + trend
        out = self.projection(seasonal) + trend_acc
        last = target[:, -8:, 0]
        if last.shape[1] < 8:
            last = F.pad(last, (8 - last.shape[1], 0))
        out = out[:, -self.pred_len:, :] + self.level_head(last).unsqueeze(-1)
        return out

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
