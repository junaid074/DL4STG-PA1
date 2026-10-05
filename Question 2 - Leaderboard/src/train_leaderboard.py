# Task 2 trainer: Autoformer with last-week holdout, calendar features, RMSE-aware loss.
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoformer import Autoformer, count_parameters

DATA = ROOT / "Data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
PRED_LEN = 168
CONT = ["feature_A", "feature_B", "feature_C", "feature_D", "feature_E", "feature_F"]
BIN = ["feature_G", "feature_H", "feature_I", "feature_J"]
ALL = CONT + BIN


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def calendar_features(n: int) -> np.ndarray:
    """Hour / day-of-week sin-cos from 1-based hourly time_idx."""
    t = np.arange(1, n + 1, dtype=np.float64)
    hour = (t - 1) % 24
    dow = ((t - 1) // 24) % 7
    feats = np.stack([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7),
        np.cos(2 * np.pi * dow / 7),
    ], axis=1).astype(np.float32)
    return feats


class WindowDataset(Dataset):
    def __init__(self, target, cov, indices, seq_len, label_len):
        self.target = target.astype(np.float32)
        self.cov = None if cov is None else cov.astype(np.float32)
        self.indices = list(indices)
        self.seq_len, self.label_len = seq_len, label_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        t = self.indices[i]
        y_enc = self.target[t - self.seq_len:t, None]
        y_dec = np.concatenate([
            self.target[t - self.label_len:t, None],
            np.zeros((PRED_LEN, 1), np.float32),
        ], 0)
        y_true = self.target[t:t + PRED_LEN, None]
        if self.cov is not None:
            x_enc = np.concatenate([y_enc, self.cov[t - self.seq_len:t]], 1)
            x_dec = np.concatenate([y_dec, self.cov[t - self.label_len:t + PRED_LEN]], 1)
        else:
            x_enc, x_dec = y_enc, y_dec

        mu = float(y_enc.mean())
        sigma = float(y_enc.std() + 1e-5)
        x_enc, x_dec = x_enc.copy(), x_dec.copy()
        x_enc[:, 0] = (x_enc[:, 0] - mu) / sigma
        x_dec[:self.label_len, 0] = (x_dec[:self.label_len, 0] - mu) / sigma
        y_true_n = (y_true - mu) / sigma
        return (torch.from_numpy(x_enc), torch.from_numpy(x_dec),
                torch.from_numpy(y_true_n), torch.tensor([mu, sigma]))


def metrics(yt, yp):
    yt, yp = np.asarray(yt, float).ravel(), np.asarray(yp, float).ravel()
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(math.sqrt(np.mean((yt - yp) ** 2)))
    smape = float(100 * np.mean(2 * np.abs(yp - yt) / np.maximum(np.abs(yt) + np.abs(yp), 1e-8)))
    return dict(MAE=mae, RMSE=rmse, sMAPE=smape)


def load_data(use_log: bool, use_ext: bool, use_cal: bool, holdout: int):
    train = pd.read_csv(DATA / "student_train.csv")
    ext = pd.read_csv(DATA / "optional_external_data.csv")
    y_raw = train["value"].to_numpy(float)
    n_full = len(y_raw)
    n_usable = n_full - holdout  # last `holdout` steps reserved for proxy-test / unused in fit

    y = y_raw.copy()
    if use_log:
        y = np.log1p(np.clip(y, 0, None))

    # covariates for train + future 168
    n_tot = n_full + PRED_LEN
    parts = []
    if use_ext:
        cov = ext[ALL].to_numpy(float)
        mu = cov[:n_usable, :len(CONT)].mean(0)
        sd = cov[:n_usable, :len(CONT)].std(0) + 1e-5
        cov = cov.copy()
        cov[:, :len(CONT)] = (cov[:, :len(CONT)] - mu) / sd
        parts.append(cov[:n_tot].astype(np.float32))
    if use_cal:
        parts.append(calendar_features(n_tot))
    cov_all = None if not parts else np.concatenate(parts, axis=1)

    y_full = np.concatenate([y, np.full(PRED_LEN, np.nan)])
    return y_full, y_raw, cov_all, use_log, n_full, n_usable


def make_origins(n_end, seq_len, stride, val_blocks, min_t=None):
    """Origins with known targets in [0, n_end)."""
    last = n_end - PRED_LEN
    first = seq_len if min_t is None else max(seq_len, min_t)
    val_start = max(first, last - val_blocks * PRED_LEN)
    train_idx = list(range(first, val_start, stride))
    val_idx = list(range(val_start, last + 1, PRED_LEN))
    return train_idx, val_idx


@torch.no_grad()
def evaluate(model, loader, device, use_log):
    model.eval()
    ys, ps = [], []
    for x_enc, x_dec, y_n, stats in loader:
        pred_n = model(x_enc.to(device), x_dec.to(device)).cpu().numpy()
        mu = stats[:, 0].numpy()[:, None, None]
        sigma = stats[:, 1].numpy()[:, None, None]
        pred = pred_n * sigma + mu
        true = y_n.numpy() * sigma + mu
        if use_log:
            pred, true = np.expm1(pred), np.expm1(true)
        pred = np.clip(pred, 0, None)
        ys.append(true)
        ps.append(pred)
    y, p = np.concatenate(ys), np.concatenate(ps)
    return metrics(y, p), y, p


@torch.no_grad()
def forecast_at(model, y_full, cov, seq_len, label_len, device, use_log, t):
    model.eval()
    y_enc = y_full[t - seq_len:t].astype(np.float32)[:, None]
    y_dec = np.concatenate([
        y_full[t - label_len:t].astype(np.float32)[:, None],
        np.zeros((PRED_LEN, 1), np.float32),
    ], 0)
    if cov is not None:
        x_enc = np.concatenate([y_enc, cov[t - seq_len:t]], 1)
        x_dec = np.concatenate([y_dec, cov[t - label_len:t + PRED_LEN]], 1)
    else:
        x_enc, x_dec = y_enc, y_dec
    mu, sigma = float(y_enc.mean()), float(y_enc.std() + 1e-5)
    x_enc, x_dec = x_enc.copy(), x_dec.copy()
    x_enc[:, 0] = (x_enc[:, 0] - mu) / sigma
    x_dec[:label_len, 0] = (x_dec[:label_len, 0] - mu) / sigma
    pred = model(torch.from_numpy(x_enc)[None].to(device),
                 torch.from_numpy(x_dec)[None].to(device)).cpu().numpy()[0, :, 0]
    pred = pred * sigma + mu
    if use_log:
        pred = np.expm1(pred)
    return np.clip(pred, 0, None)


def train_one(args):
    set_seed(args.seed)
    device = torch.device("cpu")
    # holdout=168: never train on last week; use it as proxy board score
    holdout = PRED_LEN if args.proxy_holdout else 0
    y_full, y_raw, cov, use_log, n_full, n_usable = load_data(
        args.use_log, args.use_ext, args.use_cal, holdout=holdout
    )

    # fit only on data before holdout
    train_idx, val_idx = make_origins(
        n_usable, args.seq_len, args.stride, args.val_blocks,
        min_t=args.min_train_t,
    )
    y_hist = y_full[:n_usable]

    ds_tr = WindowDataset(y_hist, cov, train_idx, args.seq_len, args.label_len)
    ds_va = WindowDataset(y_hist, cov, val_idx, args.seq_len, args.label_len)
    enc_in = 1 + (0 if cov is None else cov.shape[1])

    model = Autoformer(
        enc_in=enc_in, dec_in=enc_in, c_out=1,
        seq_len=args.seq_len, label_len=args.label_len, pred_len=PRED_LEN,
        d_model=args.d_model, n_heads=args.n_heads,
        e_layers=args.e_layers, d_layers=args.d_layers,
        moving_avg=args.moving_avg, factor=args.factor, dropout=args.dropout,
    ).to(device)
    P = count_parameters(model)
    print(f"seed={args.seed} d={args.d_model} e={args.e_layers} P={P} "
          f"train={len(ds_tr)} val={len(ds_va)} enc_in={enc_in} "
          f"log={use_log} ext={args.use_ext} cal={args.use_cal}", flush=True)

    loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(ds_va, batch_size=min(16, max(1, len(ds_va))), shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = max(1, args.epochs * len(loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.2
    )

    best, best_state, history, bad = float("inf"), None, [], 0
    epochs_run = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x_enc, x_dec, y_n, stats in loader:
            x_enc = x_enc.to(device)
            x_dec = x_dec.to(device)
            y_n = y_n.to(device)
            pred = model(x_enc, x_dec)
            # mix normalized SmoothL1 + denorm MSE (RMSE-aligned)
            loss_n = nn.functional.smooth_l1_loss(pred, y_n)
            mu = stats[:, 0].to(device).view(-1, 1, 1)
            sigma = stats[:, 1].to(device).view(-1, 1, 1)
            pred_d = pred * sigma + mu
            true_d = y_n * sigma + mu
            if use_log:
                pred_d = torch.expm1(pred_d.clamp(max=20))
                true_d = torch.expm1(true_d.clamp(max=20))
            loss_d = nn.functional.mse_loss(pred_d, true_d) / (true_d.var() + 1.0)
            loss = loss_n + args.mse_w * loss_d
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
        vm, _, _ = evaluate(model, val_loader, device, use_log)
        epochs_run = epoch
        history.append({"epoch": epoch, "train": float(np.mean(losses)), **vm})
        print(f"  ep{epoch:02d} loss={np.mean(losses):.4f} valRMSE={vm['RMSE']:.2f} "
              f"MAE={vm['MAE']:.2f}", flush=True)
        if vm["RMSE"] < best - 0.05:
            best = vm["RMSE"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if args.patience and bad >= args.patience:
                print(f"  early stop at ep{epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    E = epochs_run

    # optional refit on all usable windows (still excluding true holdout)
    if args.refit_epochs > 0:
        all_idx = list(range(args.seq_len, n_usable - PRED_LEN + 1, args.stride))
        if args.min_train_t:
            all_idx = [t for t in all_idx if t >= args.min_train_t]
        ds_all = WindowDataset(y_hist, cov, all_idx, args.seq_len, args.label_len)
        loader_all = DataLoader(ds_all, batch_size=args.batch_size, shuffle=True, drop_last=True)
        opt2 = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.25, weight_decay=args.wd)
        for ep in range(args.refit_epochs):
            model.train()
            for x_enc, x_dec, y_n, _ in loader_all:
                pred = model(x_enc.to(device), x_dec.to(device))
                loss = nn.functional.smooth_l1_loss(pred, y_n.to(device))
                opt2.zero_grad(set_to_none=True)
                loss.backward()
                opt2.step()
            E += 1
            print(f"  refit ep{ep+1}", flush=True)

    vm, _, _ = evaluate(model, val_loader, device, use_log)

    # proxy test = last 168 of original train (only if held out)
    proxy = None
    if holdout:
        pred_proxy = forecast_at(
            model, y_full, cov, args.seq_len, args.label_len, device, use_log, n_usable
        )
        proxy = metrics(y_raw[n_usable:n_full], pred_proxy)
        print(f"  PROXY last168 RMSE={proxy['RMSE']:.2f} MAE={proxy['MAE']:.2f} "
              f"mean_p={pred_proxy.mean():.1f} mean_t={y_raw[n_usable:n_full].mean():.1f}",
              flush=True)

    # final board forecast from end of full train (may retrain quickly if holdout used)
    if holdout and args.final_refit:
        # short continue on data including previously held-out week
        y_full2, y_raw2, cov2, _, n_full2, _ = load_data(
            args.use_log, args.use_ext, args.use_cal, holdout=0
        )
        all_idx = list(range(args.seq_len, n_full2 - PRED_LEN + 1, max(args.stride, 12)))
        if args.min_train_t:
            all_idx = [t for t in all_idx if t >= args.min_train_t]
        ds_all = WindowDataset(y_full2[:n_full2], cov2, all_idx, args.seq_len, args.label_len)
        loader_all = DataLoader(ds_all, batch_size=args.batch_size, shuffle=True, drop_last=True)
        opt3 = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.2, weight_decay=args.wd)
        for ep in range(args.final_refit):
            model.train()
            for x_enc, x_dec, y_n, _ in loader_all:
                pred = model(x_enc.to(device), x_dec.to(device))
                loss = nn.functional.smooth_l1_loss(pred, y_n.to(device))
                opt3.zero_grad(set_to_none=True)
                loss.backward()
                opt3.step()
            E += 1
            print(f"  final-refit ep{ep+1}", flush=True)
        y_full, cov = y_full2, cov2
        n_full = n_full2

    pred = forecast_at(
        model, y_full, cov, args.seq_len, args.label_len, device, use_log, n_full
    )

    # light bias calibrate toward recent level (helps distribution shift)
    if args.calibrate > 0:
        recent = y_raw[-PRED_LEN:]
        scale = (recent.mean() + 1e-3) / (pred.mean() + 1e-3)
        scale = float(np.clip(scale, 1.0 - args.calibrate, 1.0 + args.calibrate))
        pred = np.clip(pred * scale, 0, None)
        print(f"  calibrate scale={scale:.3f} -> mean={pred.mean():.1f}", flush=True)

    tag = (
        f"d{args.d_model}_e{args.e_layers}_s{args.seed}"
        f"{'_log' if use_log else '_nolog'}"
        f"{'_ext' if args.use_ext else ''}"
        f"{'_cal' if args.use_cal else ''}"
        f"_ep{args.epochs}"
    )
    if args.min_train_t:
        tag += f"_mt{args.min_train_t}"
    if args.refit_epochs:
        tag += f"_rf{args.refit_epochs}"
    if args.final_refit:
        tag += f"_fr{args.final_refit}"

    result = {
        "tag": tag, "seed": args.seed, "params": P, "epochs": E,
        "best_val_RMSE": best, "final_val": vm, "proxy": proxy,
        "forecast": pred.tolist(), "forecast_mean": float(pred.mean()),
        "history": history, "config": vars(args),
    }
    path = OUT / f"run_{tag}.json"
    path.write_text(json.dumps(result))
    (OUT / f"pred_{tag}.txt").write_text(",".join(f"{x:.6f}" for x in pred) + "\n")
    proxy_s = f" proxy={proxy['RMSE']:.2f}" if proxy else ""
    print(f"WROTE {path} val={best:.2f}{proxy_s} P={P} E={E} mean={pred.mean():.1f}",
          flush=True)
    return result


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-log", action="store_true", default=True)
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--use-ext", action="store_true", default=True)
    p.add_argument("--no-ext", action="store_true")
    p.add_argument("--use-cal", action="store_true", default=True)
    p.add_argument("--no-cal", action="store_true")
    p.add_argument("--seq-len", type=int, default=336)
    p.add_argument("--label-len", type=int, default=84)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--e-layers", type=int, default=2)
    p.add_argument("--d-layers", type=int, default=1)
    p.add_argument("--moving-avg", type=int, default=25)
    p.add_argument("--factor", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--refit-epochs", type=int, default=0)
    p.add_argument("--final-refit", type=int, default=2)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--mse-w", type=float, default=0.15)
    p.add_argument("--stride", type=int, default=24)
    p.add_argument("--val-blocks", type=int, default=6)
    p.add_argument("--min-train-t", type=int, default=0,
                   help="only use origins >= this (recent-data focus)")
    p.add_argument("--proxy-holdout", action="store_true", default=True)
    p.add_argument("--no-proxy-holdout", action="store_true")
    p.add_argument("--calibrate", type=float, default=0.25,
                   help="max relative scale toward last-week mean")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.no_log:
        args.use_log = False
    if args.no_ext:
        args.use_ext = False
    if args.no_cal:
        args.use_cal = False
    if args.no_proxy_holdout:
        args.proxy_holdout = False
    train_one(args)
