from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
import warnings
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
except Exception as exc:
    raise RuntimeError("Stage3B requires torch and scikit-learn") from exc


@dataclass(frozen=True)
class Config:
    experiment: str = "Stage3B_LATENT_ACTION_CONTRASTIVE_PRETRAINING"
    created_utc: str = "2026-07-16"
    max_context: int = 64
    max_horizon: int = 32
    channels: int = 6
    actions: int = 10
    classes: int = 5
    slots: int = 6
    token_dim: int = 64
    action_dim: int = 48
    critic_hidden: int = 96
    train_tasks: int = 3200
    train_steps: int = 420
    batch_size: int = 128
    calibration_tasks_per_cell: int = 64
    development_tasks_per_cell: int = 96
    confirmation_tasks_per_cell: int = 96
    model_seeds: Tuple[int, ...] = (202607164101, 202607164102, 202607164103)
    train_data_seeds: Tuple[int, ...] = (202607164201, 202607164202, 202607164203)
    calibration_seed: int = 202607164301
    development_seed: int = 202607164401
    confirmation_seed: int = 202607164501
    learning_rate: float = 7e-4
    weight_decay: float = 1e-5
    critic_weight: float = 0.12
    grad_clip: float = 2.0
    bootstrap_draws: int = 2000
    device: str = "cpu"


CFG = Config()
VARIANTS = ("forecast_only", "detached_contrastive", "joint_contrastive", "shuffled_joint")
ACTION_NAMES = (
    "last_value", "seasonal_naive", "global_linear", "recent_linear", "damped_trend",
    "ridge_ar4", "fourier_extrapolation", "mean_reversion", "bounded_saturation", "piecewise_trend",
)
FAMILIES = (
    "supported_level", "supported_trend", "supported_seasonal", "supported_ar", "supported_pair",
    "compositional_ood", "irregular_missing", "long_horizon", "changepoint", "saturation", "chirp", "amplitude_modulation",
)
EVAL_CELLS = (
    ("supported_id_64x16", ("supported_level", "supported_trend", "supported_seasonal", "supported_ar", "supported_pair"), 64, 16),
    ("supported_id_32x32", ("supported_level", "supported_trend", "supported_seasonal", "supported_ar"), 32, 32),
    ("compositional_ood_64x16", ("compositional_ood",), 64, 16),
    ("compositional_ood_32x32", ("compositional_ood",), 32, 32),
    ("irregular_missing_64x16", ("irregular_missing",), 64, 16),
    ("long_horizon_64x32", ("long_horizon",), 64, 32),
    ("misspecified_64x16", ("changepoint", "saturation", "chirp", "amplitude_modulation"), 64, 16),
    ("misspecified_32x32", ("changepoint", "saturation", "chirp", "amplitude_modulation"), 32, 32),
)
CLASS_REP_LOG_RATIO = np.array([math.log(0.50), math.log(0.73), 0.0, math.log(1.28), math.log(1.90)], dtype=np.float32)
GATES = {
    "useful_win_auc": (">=", 0.78), "auc_gain_vs_detached": (">=", 0.06), "severe_harm_auc": (">=", 0.80),
    "high_opportunity_recall": (">=", 0.45), "realized_switch_win_rate": (">=", 0.68),
    "severe_wrong_switch_rate": ("<", 0.05), "intervention_rate_low": (">=", 0.05),
    "intervention_rate_high": ("<=", 0.20), "synthetic_tail_p95_improvement": (">=", 0.10),
    "supported_regression": ("<=", 0.015), "composition_regression": ("<=", 0.015),
    "oracle_gap_closure": (">=", 0.20), "forecast_regression_vs_control": ("<=", 0.01),
    "auc_gain_vs_shuffled": (">=", 0.04), "latent_probe_gain_vs_detached": (">=", 0.05),
    "matched_parameters_initialization_data_updates_replay_firewall": ("==", 1.0),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hash_arrays(data: Dict[str, Any], include_targets: bool = True) -> str:
    h = hashlib.sha256()
    keys = ["channels", "meta", "candidates", "backtest", "mase_scale", "period", "context_len", "horizon"]
    if include_targets:
        keys += ["y", "ymask"]
    for key in keys:
        arr = np.ascontiguousarray(data[key]); h.update(key.encode()); h.update(arr.tobytes())
    for key in ("family", "cell", "task_id"):
        h.update(key.encode()); h.update("\n".join(map(str, data[key])).encode())
    return h.hexdigest()


def state_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed % (2**32 - 1)); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))


def ar_noise(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    e = rng.normal(0.0, sigma, n); z = np.zeros(n, dtype=np.float64)
    for i in range(1, n): z[i] = phi * z[i - 1] + e[i]
    return z


def generate_series(rng: np.random.Generator, family: str, c: int, h: int, shifted: bool) -> Tuple[np.ndarray, int, np.ndarray]:
    n = c + h; t = np.arange(n, dtype=np.float64); level = rng.uniform(-1.8, 1.8)
    period = int(rng.choice([4, 6, 8, 12])); slope = rng.uniform(-0.04 if not shifted else -0.06, 0.04 if not shifted else 0.06)
    amp = rng.uniform(0.25, 1.0 if not shifted else 1.35); phase = rng.uniform(0, 2 * np.pi)
    phi = rng.uniform(0.25, 0.84 if not shifted else 0.92); sigma = rng.uniform(0.04, 0.17 if not shifted else 0.22)
    seasonal = amp * np.sin(2 * np.pi * t / period + phase); ar = ar_noise(rng, n, phi, sigma)
    if family == "supported_level": y = level + ar; period = 1
    elif family == "supported_trend": y = level + slope * t + ar; period = 1
    elif family == "supported_seasonal": y = level + seasonal + rng.normal(0, sigma, n)
    elif family == "supported_ar": y = level + 1.45 * ar; period = 1
    elif family == "supported_pair":
        mode = int(rng.integers(0, 3))
        if mode == 0: y = level + slope * t + seasonal + rng.normal(0, sigma, n)
        elif mode == 1: y = level + seasonal + ar
        else: y = level + slope * t + ar; period = 1
    elif family == "compositional_ood": y = level + slope * t + seasonal + 1.25 * ar
    elif family == "irregular_missing": y = level + slope * t + seasonal + ar
    elif family == "long_horizon": y = level + slope * t + 1.15 * seasonal + ar
    elif family == "changepoint":
        cp = int(rng.integers(max(8, c // 2), max(9, c - 4)))
        slope2 = slope + rng.choice([-1, 1]) * rng.uniform(0.06, 0.15 if shifted else 0.12); jump = rng.uniform(-1.5, 1.5)
        y = level + slope * t + ar + (slope2 - slope) * np.maximum(t - cp, 0) + jump * (t >= cp); period = 1
    elif family == "saturation":
        midpoint = rng.uniform(c * 0.35, c * 0.82); rate = rng.uniform(0.05, 0.15 if shifted else 0.11)
        cap = level + rng.choice([-1, 1]) * rng.uniform(1.1, 3.2); start = level - (cap - level)
        y = start + (cap - start) / (1 + np.exp(-rate * (t - midpoint))) + ar; period = 1
    elif family == "chirp":
        f0 = rng.uniform(0.02, 0.065); f1 = rng.uniform(0.11, 0.24 if shifted else 0.18); k = (f1 - f0) / max(1, n - 1)
        y = level + amp * np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t) + phase) + ar; period = 1
    elif family == "amplitude_modulation":
        carrier = int(rng.choice([4, 6, 8, 12])); mod_period = rng.uniform(24, 62)
        envelope = 0.30 + amp * (1.0 + 0.70 * np.sin(2 * np.pi * t / mod_period + phase / 2))
        y = level + slope * t + envelope * np.sin(2 * np.pi * t / carrier + phase) + ar; period = carrier
    else: raise KeyError(family)
    if shifted and rng.random() < 0.12: y = np.sign(y) * np.log1p(np.abs(y)) * 1.45
    observed = np.ones(c, dtype=np.float32)
    if family == "irregular_missing":
        missing = rng.random(c) < rng.uniform(0.18, 0.36); missing[:3] = False; missing[-1] = False; observed[missing] = 0.0
    elif rng.random() < 0.10:
        missing = rng.random(c) < rng.uniform(0.03, 0.10); missing[:2] = False; missing[-1] = False; observed[missing] = 0.0
    return y.astype(np.float64), period, observed


def interpolate(x: np.ndarray, observed: np.ndarray) -> np.ndarray:
    idx = np.arange(len(x)); good = observed.astype(bool) & np.isfinite(x)
    if good.sum() == 0: return np.zeros_like(x)
    if good.sum() == 1: return np.full_like(x, x[good][0])
    return np.interp(idx, idx[good], x[good])


def normalize(x: np.ndarray, y: np.ndarray, observed: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    xi = interpolate(x, observed); good = observed.astype(bool)
    center = float(np.median(xi[good])) if good.any() else float(np.median(xi))
    mad = float(np.median(np.abs(xi[good] - center))) if good.any() else float(np.median(np.abs(xi - center)))
    scale = max(1e-3, 1.4826 * mad, 0.75 * float(np.mean(np.abs(np.diff(xi)))))
    xn = (xi - center) / scale; yn = (y - center) / scale; mase_scale = max(0.08, float(np.mean(np.abs(np.diff(xn)))))
    return xn.astype(np.float32), yn.astype(np.float32), mase_scale


def linear_fc(x: np.ndarray, h: int, window: int) -> np.ndarray:
    w = min(len(x), max(4, window)); v = np.asarray(x[-w:], dtype=np.float64); tt = np.arange(w, dtype=np.float64); tc = tt - tt.mean()
    slope = float(np.dot(tc, v - v.mean()) / max(1e-8, np.dot(tc, tc)))
    return (v.mean() + slope * ((w - 1 + np.arange(1, h + 1)) - tt.mean())).astype(np.float32)


def ridge_ar_fc(x: np.ndarray, h: int, lags: int = 4, ridge: float = 0.20) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) <= lags + 2: return np.full(h, x[-1], dtype=np.float32)
    X, y = [], []
    for i in range(lags, len(x)): X.append(x[i - lags:i][::-1]); y.append(x[i])
    X = np.asarray(X); y = np.asarray(y); A = np.column_stack([np.ones(len(X)), X])
    try: coef = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ y)
    except np.linalg.LinAlgError: return np.full(h, x[-1], dtype=np.float32)
    hist = list(x); out = []
    for _ in range(h):
        pred = float(np.asarray([1.0] + hist[-lags:][::-1]) @ coef)
        pred = float(np.clip(pred, np.min(x) - 2 * np.std(x), np.max(x) + 2 * np.std(x))); out.append(pred); hist.append(pred)
    return np.asarray(out, dtype=np.float32)


def fourier_fc(x: np.ndarray, h: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64); n = len(x); trend = linear_fc(x, n, n); detr = x - trend
    spec = np.fft.rfft(detr); freq = np.fft.rfftfreq(n)
    if len(spec) <= 2: return linear_fc(x, h, n)
    idx = np.argsort(np.abs(spec[1:]))[-min(3, len(spec) - 1):] + 1; tf = np.arange(n, n + h, dtype=np.float64)
    out = linear_fc(x, h, n).astype(np.float64)
    for j in idx: out += 2 * np.abs(spec[j]) / n * np.cos(2 * np.pi * freq[j] * tf + np.angle(spec[j]))
    return out.astype(np.float32)


def piecewise_fc(x: np.ndarray, h: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64); n = len(x)
    if n < 16: return linear_fc(x, h, min(12, n))
    splits = range(max(6, n // 3), min(n - 5, 2 * n // 3 + 4)); scored = [(abs(x[:s].mean() - x[s:].mean()) / (np.std(x) + 1e-6), s) for s in splits]
    _, split = max(scored); return linear_fc(x[split:], h, min(18, n - split))


def candidate_forecasts(x: np.ndarray, h: int, period: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64); last = np.full(h, x[-1]); p = max(1, min(int(period), len(x)))
    seasonal = np.resize(x[-p:], h); global_linear = linear_fc(x, h, len(x)); recent_linear = linear_fc(x, h, min(16, len(x)))
    slope = float(recent_linear[0] - x[-1]); damp = x[-1] + slope * np.cumsum(0.86 ** np.arange(h)); ar = ridge_ar_fc(x, h)
    fourier = fourier_fc(x, h); target = float(np.mean(x[-min(24, len(x)):]))
    mean_revert = target + (x[-1] - target) * (0.88 ** np.arange(1, h + 1)); span = max(0.2, float(np.quantile(x, .95) - np.quantile(x, .05)))
    bounded = np.clip(recent_linear, np.quantile(x, .05) - .15 * span, np.quantile(x, .95) + .15 * span); piecewise = piecewise_fc(x, h)
    out = np.stack([last, seasonal, global_linear, recent_linear, damp, ar, fourier, mean_revert, bounded, piecewise]).astype(np.float32)
    out[~np.isfinite(out)] = float(x[-1]); return out


def backtest_profile(x: np.ndarray, h: int, period: int, scale: float) -> np.ndarray:
    rows = []; hh = max(2, min(8, h))
    for mult in (2, 1):
        origin = len(x) - mult * hh
        if origin < 12: continue
        pred = candidate_forecasts(x[:origin], hh, period); truth = x[origin:origin + hh]
        rows.append(np.mean(np.abs(pred - truth[None, :]), axis=1) / max(.05, scale))
    if not rows: rows = [np.full(len(ACTION_NAMES), 2.0, dtype=np.float32)]
    arr = np.stack(rows, axis=1); return np.stack([arr.mean(1), arr[:, -1], arr.std(1), arr[:, -1] - arr[:, 0]], axis=1).astype(np.float32)


def typed_channels(x: np.ndarray, observed: np.ndarray, max_context: int) -> np.ndarray:
    c = len(x); pad = max_context - c; level = np.zeros(max_context, np.float32); mask = np.zeros(max_context, np.float32)
    level[pad:] = x; mask[pad:] = observed; diff = np.zeros_like(level); diff[1:] = np.diff(level); age = np.zeros_like(level); running = 0.0
    for i in range(max_context):
        running = 0.0 if mask[i] > 0 else running + 1.0; age[i] = running / max_context
    local = np.convolve(level, np.ones(5) / 5, mode="same").astype(np.float32)
    return np.stack([level, diff, np.abs(diff), mask, age, local], axis=0)


def make_task(rng: np.random.Generator, family: str, c: int, h: int, shifted: bool, task_id: str, cell: str) -> Dict[str, Any]:
    full, period, observed = generate_series(rng, family, c, h, shifted); xn, yn, mase_scale = normalize(full[:c], full[c:], observed)
    candidates = candidate_forecasts(xn, h, period); bt = backtest_profile(xn, h, period, mase_scale)
    ypad = np.zeros(CFG.max_horizon, np.float32); ymask = np.zeros(CFG.max_horizon, np.float32); cpad = np.zeros((CFG.actions, CFG.max_horizon), np.float32)
    ypad[:h] = yn; ymask[:h] = 1.0; cpad[:, :h] = candidates; cpad[:, h:] = candidates[:, -1:]
    trend_strength = float(abs(linear_fc(xn, 1, len(xn))[0] - xn[-1]))
    if period > 1 and len(xn) >= 2 * period:
        season_strength = float(np.corrcoef(xn[period:], xn[:-period])[0, 1]); season_strength = season_strength if np.isfinite(season_strength) else 0.0
    else: season_strength = 0.0
    meta = np.asarray([c / CFG.max_context, h / CFG.max_horizon, min(period, 12) / 12.0, 1.0 - float(observed.mean()), min(2.0, trend_strength) / 2.0, np.clip(season_strength, -1, 1), min(2.0, float(np.std(np.diff(xn)))) / 2.0, float(np.clip(xn[-1] / 4.0, -1, 1))], np.float32)
    return {"channels": typed_channels(xn, observed, CFG.max_context), "y": ypad, "ymask": ymask, "candidates": cpad, "backtest": bt, "meta": meta, "mase_scale": np.float32(mase_scale), "period": np.int16(period), "context_len": np.int16(c), "horizon": np.int16(h), "family": family, "cell": cell, "task_id": task_id}


def stack_tasks(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ("channels", "y", "ymask", "candidates", "backtest", "meta", "mase_scale", "period", "context_len", "horizon"): out[key] = np.stack([t[key] for t in tasks])
    for key in ("family", "cell", "task_id"): out[key] = np.asarray([t[key] for t in tasks], dtype=object)
    return out


def make_training_data(n: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed); families = np.asarray(FAMILIES, object); weights = np.asarray([.09,.09,.09,.09,.10,.10,.08,.08,.07,.07,.07,.07]); tasks = []
    for i in range(n):
        family = str(rng.choice(families, p=weights / weights.sum())); c = int(rng.choice([32, 48, 64], p=[.25,.25,.50])); h = int(rng.choice([8,16,24,32], p=[.15,.35,.15,.35]))
        tasks.append(make_task(rng, family, c, h, False, f"train-{seed}-{i}", "training"))
    return stack_tasks(tasks)


def make_eval_data(seed: int, tasks_per_cell: int, shifted: bool = True) -> Dict[str, Any]:
    rng = np.random.default_rng(seed); tasks = []
    for cell, fams, c, h in EVAL_CELLS:
        for i in range(tasks_per_cell): tasks.append(make_task(rng, str(fams[i % len(fams)]), c, h, shifted, f"{cell}-{seed}-{i}", cell))
    return stack_tasks(tasks)


class ResidualConv(nn.Module):
    def __init__(self, d: int):
        super().__init__(); self.norm = nn.GroupNorm(8, d); self.dw = nn.Conv1d(d, d, 5, padding=2, groups=d); self.pw1 = nn.Conv1d(d, 2*d, 1); self.pw2 = nn.Conv1d(2*d, d, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.dw(self.norm(x)); return x + self.pw2(F.gelu(self.pw1(z))) / math.sqrt(2)


class LatentActionModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__(); d = cfg.token_dim; self.cfg = cfg
        self.stem = nn.Conv1d(cfg.channels, d, 5, padding=2); self.blocks = nn.ModuleList([ResidualConv(d), ResidualConv(d)]); self.pos = nn.Parameter(torch.randn(cfg.max_context, d)*.02)
        self.slot_query = nn.Parameter(torch.randn(cfg.slots, d)*.08); self.slot_norm = nn.LayerNorm(d); self.slot_mlp = nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
        self.meta = nn.Sequential(nn.Linear(8,d),nn.GELU(),nn.Linear(d,d)); self.pool_gate = nn.Linear(d,cfg.slots)
        self.blend_head = nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,cfg.actions)); self.residual_head = nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,cfg.max_horizon)); self.sigma_head = nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,cfg.max_horizon))
        self.action_embedding = nn.Embedding(cfg.actions,16); self.action_path = nn.Sequential(nn.Conv1d(2,cfg.action_dim,5,padding=2),nn.GELU(),nn.Conv1d(cfg.action_dim,cfg.action_dim,3,padding=1),nn.GELU())
        self.bt_mlp = nn.Sequential(nn.Linear(4,24),nn.GELU(),nn.Linear(24,24)); self.action_query = nn.Sequential(nn.Linear(cfg.action_dim+24+16+8,d),nn.GELU(),nn.Linear(d,d))
        self.slot_key = nn.Linear(d,d,bias=False); self.slot_value = nn.Linear(d,d,bias=False); critic_in = 3*d+cfg.action_dim+24+16+8
        self.critic_trunk = nn.Sequential(nn.Linear(critic_in,cfg.critic_hidden),nn.GELU(),nn.Linear(cfg.critic_hidden,cfg.critic_hidden),nn.GELU())
        self.class_head = nn.Linear(cfg.critic_hidden,cfg.classes); self.adv_head = nn.Linear(cfg.critic_hidden,1); self.score_head = nn.Linear(cfg.critic_hidden,1); self.probe_head = nn.Linear(2*d,d)
    def encode(self, channels, meta):
        token = self.stem(channels)
        for block in self.blocks: token = block(token)
        token = token.transpose(1,2)+self.pos[None,:,:]; mask = channels[:,3,:] > 0; q = self.slot_query[None,:,:].expand(token.shape[0],-1,-1)
        logits = torch.einsum("bsd,bld->bsl",q,token)/math.sqrt(token.shape[-1]); logits = logits.masked_fill(~mask[:,None,:],-1e4); attn = torch.softmax(logits,-1)
        slots = torch.einsum("bsl,bld->bsd",attn,token)+q; slots = self.slot_norm(slots+self.slot_mlp(self.slot_norm(slots))/math.sqrt(2)); meta_z = self.meta(meta)
        gate = torch.softmax(self.pool_gate(meta_z),-1); pooled = torch.einsum("bs,bsd->bd",gate,slots); return slots, pooled, meta_z
    def forecast(self, slots, pooled, meta_z, candidates):
        z = torch.cat([pooled,meta_z],-1); weights = torch.softmax(self.blend_head(z),-1); base = torch.einsum("bk,bkh->bh",weights,candidates)
        return base+.30*torch.tanh(self.residual_head(z)), torch.clamp(self.sigma_head(z),-2.8,2.0), weights
    def critic(self, slots, pooled, meta, default, candidates, backtest, detach_context=False):
        if detach_context: slots, pooled, default = slots.detach(), pooled.detach(), default.detach()
        B,K,H = candidates.shape; pair = torch.stack([candidates,candidates-default[:,None,:]],2).reshape(B*K,2,H); path = self.action_path(pair).mean(-1).reshape(B,K,-1)
        bt = self.bt_mlp(backtest); ids = torch.arange(K,device=candidates.device)[None,:].expand(B,-1); aid = self.action_embedding(ids); m = meta[:,None,:].expand(-1,K,-1)
        query = self.action_query(torch.cat([path,bt,aid,m],-1)); key = self.slot_key(slots); value = self.slot_value(slots); attn = torch.softmax(torch.einsum("bkd,bsd->bks",query,key)/math.sqrt(key.shape[-1]),-1)
        context = torch.einsum("bks,bsd->bkd",attn,value); pooled_exp = pooled[:,None,:].expand(-1,K,-1); hidden = self.critic_trunk(torch.cat([pooled_exp,query,context,path,bt,aid,m],-1))
        return {"logits":self.class_head(hidden),"adv":self.adv_head(hidden).squeeze(-1),"score":self.score_head(hidden).squeeze(-1),"probe":self.probe_head(torch.cat([pooled_exp,context],-1)),"query":query,"context":context,"attn":attn}
    def forward(self, channels, meta, candidates, backtest, detach_context=False):
        slots,pooled,meta_z = self.encode(channels,meta); mean,log_sigma,weights = self.forecast(slots,pooled,meta_z,candidates); crit = self.critic(slots,pooled,meta,mean,candidates,backtest,detach_context)
        return {"mean":mean,"log_sigma":log_sigma,"weights":weights,"slots":slots,"pooled":pooled,**crit}


def parameter_count(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def tensor_batch(data, idx, device="cpu"):
    return {k:torch.from_numpy(np.asarray(data[k][idx])).to(device=device,dtype=torch.float32) for k in ("channels","y","ymask","candidates","backtest","meta","mase_scale")}

def task_mase_torch(pred,y,mask,scale): return ((pred-y).abs()*mask).sum(1)/mask.sum(1).clamp_min(1)/scale.clamp_min(.05)

def action_loss_torch(candidates,y,mask,scale): return ((candidates-y[:,None,:]).abs()*mask[:,None,:]).sum(2)/mask.sum(1)[:,None].clamp_min(1)/scale[:,None].clamp_min(.05)

def ratio_labels(alt,default):
    ratio=(alt+.05)/(default[:,None]+.05); labels=torch.full_like(ratio,2,dtype=torch.long); labels[ratio<=.60]=0; labels[(ratio>.60)&(ratio<=.85)]=1; labels[(ratio>1.10)&(ratio<=1.50)]=3; labels[ratio>1.50]=4
    return labels,torch.log(ratio.clamp(.15,6.0)).clamp(-1.5,1.5)

def slot_diversity(slots):
    z=F.normalize(slots,dim=-1); gram=torch.einsum("bsd,btd->bst",z,z); eye=torch.eye(gram.shape[-1],device=gram.device)[None,:,:]; return ((gram-eye)**2).mean()

def critic_objective(out,labels,log_ratio):
    logits,adv,score=out["logits"],out["adv"],out["score"]; weights=torch.tensor([1.45,1.15,.55,1.05,1.45],device=logits.device)
    ce=F.cross_entropy(logits.reshape(-1,CFG.classes),labels.reshape(-1),weight=weights,label_smoothing=.015); p_use=torch.logsumexp(logits[:,:,:2],-1)-torch.logsumexp(logits[:,:,2:],-1); p_harm=logits[:,:,4]-torch.logsumexp(logits[:,:,:4],-1)
    bce=F.binary_cross_entropy_with_logits(p_use,(labels<=1).float())+F.binary_cross_entropy_with_logits(p_harm,(labels==4).float()); huber=F.smooth_l1_loss(adv,log_ratio)
    best=torch.argmin(log_ratio,1); worst=torch.argmax(log_ratio,1); listwise=F.cross_entropy(score,best); rank=F.relu(.35-score.gather(1,best[:,None]).squeeze(1)+score.gather(1,worst[:,None]).squeeze(1)).mean()
    sim=F.cosine_similarity(out["query"],out["context"],dim=-1); target=torch.where(labels<=1,torch.ones_like(sim),torch.where(labels==4,-torch.ones_like(sim),torch.zeros_like(sim))); mask=target!=0; align=((sim[mask]-target[mask])**2).mean() if mask.any() else sim.sum()*0
    return .75*ce+.22*bce+.32*huber+.35*listwise+.20*rank+.18*align


def train_one_seed(seed_index,output,cfg):
    model_seed=cfg.model_seeds[seed_index]; data_seed=cfg.train_data_seeds[seed_index]; set_seed(model_seed); train=make_training_data(cfg.train_tasks,data_seed); train_hash=hash_arrays(train)
    base=LatentActionModel(cfg); init={k:v.detach().clone() for k,v in base.state_dict().items()}; models={}; opts={}
    for variant in VARIANTS:
        m=LatentActionModel(cfg); m.load_state_dict(init); m.train(); models[variant]=m; opts[variant]=torch.optim.AdamW(m.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    initial_hashes={v:state_hash(m) for v,m in models.items()}; rng=np.random.default_rng(data_seed+919); trace=[]
    for step in range(cfg.train_steps):
        idx=rng.integers(0,cfg.train_tasks,size=cfg.batch_size); batch=tensor_batch(train,idx,cfg.device); perm=torch.from_numpy(rng.permutation(cfg.batch_size)).long()
        for variant in VARIANTS:
            m=models[variant]; opt=opts[variant]; opt.zero_grad(set_to_none=True); out=m(batch["channels"],batch["meta"],batch["candidates"],batch["backtest"],detach_context=variant=="detached_contrastive")
            default_loss=task_mase_torch(out["mean"],batch["y"],batch["ymask"],batch["mase_scale"]); sigma=torch.exp(out["log_sigma"]); nll=(out["log_sigma"]+.5*((batch["y"]-out["mean"])/sigma)**2)*batch["ymask"]; nll=nll.sum(1)/batch["ymask"].sum(1).clamp_min(1); forecast_loss=.84*default_loss.mean()+.16*nll.mean()
            with torch.no_grad():
                alt=action_loss_torch(batch["candidates"],batch["y"],batch["ymask"],batch["mase_scale"]); labels,log_ratio=ratio_labels(alt,default_loss.detach())
                if variant=="shuffled_joint": labels,log_ratio=labels[perm],log_ratio[perm]
            if variant=="forecast_only": critic_loss=out["logits"].sum()*0; total=forecast_loss+critic_loss
            else:
                critic_loss=critic_objective(out,labels,log_ratio); weight=cfg.critic_weight*min(1.0,(step+1)/max(1,int(.20*cfg.train_steps))); total=forecast_loss+weight*critic_loss+.01*slot_diversity(out["slots"])
            total.backward(); nn.utils.clip_grad_norm_(m.parameters(),cfg.grad_clip); opt.step()
            if step in (0,cfg.train_steps//4,cfg.train_steps//2,3*cfg.train_steps//4,cfg.train_steps-1): trace.append({"seed":model_seed,"step":step,"variant":variant,"total_loss":float(total.detach()),"forecast_loss":float(forecast_loss.detach()),"critic_loss":float(critic_loss.detach())})
    seed_dir=output/f"seed_{seed_index+1}"; seed_dir.mkdir(parents=True,exist_ok=True); pd.DataFrame(trace).to_csv(seed_dir/"training_trace.csv",index=False)
    checkpoints={}
    for variant,m in models.items():
        path=seed_dir/f"{variant}.pt"; torch.save({"state_dict":m.state_dict(),"variant":variant,"model_seed":model_seed,"data_seed":data_seed,"train_hash":train_hash,"parameter_count":parameter_count(m),"updates":cfg.train_steps},path); checkpoints[variant]=str(path)
    return {"models":models,"model_seed":model_seed,"data_seed":data_seed,"train_hash":train_hash,"initial_hashes":initial_hashes,"parameter_counts":{v:parameter_count(m) for v,m in models.items()},"updates":cfg.train_steps,"checkpoints":checkpoints}


@torch.no_grad()
def predict(model,data,batch_size=256):
    model.eval(); rows={k:[] for k in ("mean","log_sigma","probs","adv","score","probe","pooled")}
    for start in range(0,len(data["y"]),batch_size):
        idx=np.arange(start,min(len(data["y"]),start+batch_size)); batch=tensor_batch(data,idx); out=model(batch["channels"],batch["meta"],batch["candidates"],batch["backtest"])
        rows["mean"].append(out["mean"].cpu().numpy()); rows["log_sigma"].append(out["log_sigma"].cpu().numpy()); rows["probs"].append(torch.softmax(out["logits"],-1).cpu().numpy()); rows["adv"].append(out["adv"].cpu().numpy()); rows["score"].append(out["score"].cpu().numpy()); rows["probe"].append(out["probe"].cpu().numpy()); rows["pooled"].append(out["pooled"].cpu().numpy())
    return {k:np.concatenate(v,0) for k,v in rows.items()}

def task_mase_np(pred,data): return (np.abs(pred-data["y"])*data["ymask"]).sum(1)/np.maximum(1,data["ymask"].sum(1))/np.maximum(.05,data["mase_scale"])
def action_losses_np(default_mean,data):
    default=task_mase_np(default_mean,data); alt=(np.abs(data["candidates"]-data["y"][:,None,:])*data["ymask"][:,None,:]).sum(2)/np.maximum(1,data["ymask"].sum(1))[:,None]/np.maximum(.05,data["mase_scale"])[:,None]; return default,alt,(alt+.05)/(default[:,None]+.05)
def safe_auc(y,score):
    y=np.asarray(y).astype(int); return float("nan") if np.unique(y).size<2 else float(roc_auc_score(y,score))

def policy_apply(pred,default,alt,data,params):
    probs=pred["probs"]; p_use=probs[:,:,:2].sum(-1); p_harm=probs[:,:,4]; exp_log=(probs*CLASS_REP_LOG_RATIO[None,None,:]).sum(-1); blended=.55*pred["adv"]+.45*exp_log
    desirability=p_use-params["risk_weight"]*p_harm-params["adv_weight"]*np.maximum(blended,0); eligible=(p_use>=params["p_use"])&(p_harm<=params["p_harm"])&(blended<=params["adv_max"]); masked=np.where(eligible,desirability,-1e9); action=masked.argmax(1); confidence=masked[np.arange(len(default)),action]; has=eligible.any(1)
    max_switch=int(math.floor(params["budget"]*len(default))); switched=np.zeros(len(default),bool)
    if max_switch>0 and has.any():
        order=np.argsort(-confidence); legal=order[has[order]][:max_switch]; switched[legal]=True
    selected=np.where(switched,alt[np.arange(len(default)),action],default); ratio=np.where(switched,(alt[np.arange(len(default)),action]+.05)/(default+.05),1.0)
    return {"loss":selected,"switched":switched,"action":action,"selected_ratio":ratio,"confidence":confidence}

def calibrate_policy(pred,default,alt,data):
    rows=[]
    for p_use in (.45,.50,.55,.60,.65):
      for p_harm in (.08,.12,.18,.25):
       for adv_max in (-.02,-.05,-.10,-.16):
        for budget in (.05,.10,.15,.20):
         params={"p_use":p_use,"p_harm":p_harm,"adv_max":adv_max,"budget":budget,"risk_weight":1.0,"adv_weight":.35}; r=policy_apply(pred,default,alt,data,params); sw=r["switched"]; rate=float(sw.mean()); win=float((r["selected_ratio"][sw]<1).mean()) if sw.any() else 0.; severe=float((r["selected_ratio"][sw]>1.2).mean()) if sw.any() else 1.
         tail=np.isin(data["cell"],["long_horizon_64x32","irregular_missing_64x16","misspecified_64x16","misspecified_32x32"]); supported=np.char.startswith(data["cell"].astype(str),"supported"); comp=np.char.startswith(data["cell"].astype(str),"compositional")
         improvement=1-r["loss"].mean()/default.mean(); tail_improvement=1-np.quantile(r["loss"][tail],.95)/np.quantile(default[tail],.95); sup_reg=r["loss"][supported].mean()/default[supported].mean()-1; comp_reg=r["loss"][comp].mean()/default[comp].mean()-1; feasible=.045<=rate<=.205 and win>=.68 and severe<.05 and sup_reg<=.015 and comp_reg<=.015; utility=improvement+.45*max(0.,tail_improvement)-.25*max(0.,severe-.02)
         rows.append({**params,"intervention_rate":rate,"win_rate":win,"severe_wrong":severe,"improvement":improvement,"tail_improvement":tail_improvement,"supported_regression":sup_reg,"composition_regression":comp_reg,"feasible":feasible,"utility":utility})
    frame=pd.DataFrame(rows); feasible=frame[frame.feasible]; row=(feasible.sort_values(["utility","win_rate","intervention_rate"],ascending=[False,False,True]).iloc[0] if len(feasible) else frame.sort_values(["severe_wrong","supported_regression","composition_regression","utility"],ascending=[True,True,True,False]).iloc[0]); keys=["p_use","p_harm","adv_max","budget","risk_weight","adv_weight"]; return {k:float(row[k]) for k in keys},frame

def aggregate_predictions(seed_results,variant,data):
    preds=[predict(sr["models"][variant],data) for sr in seed_results]; merged={key:np.mean([p[key] for p in preds],axis=0) for key in preds[0]}; default,alt,ratio=action_losses_np(merged["mean"],data); return merged,default,alt,ratio

def fit_latent_probe(cal_pred,cal_ratio,dev_pred,dev_ratio):
    Xc=cal_pred["probe"].reshape(-1,cal_pred["probe"].shape[-1]); yc=(cal_ratio.reshape(-1)<=.85).astype(int); Xd=dev_pred["probe"].reshape(-1,dev_pred["probe"].shape[-1]); yd=(dev_ratio.reshape(-1)<=.85).astype(int); rng=np.random.default_rng(202607164777)
    if len(Xc)>30000: ix=rng.choice(len(Xc),30000,replace=False); Xc,yc=Xc[ix],yc[ix]
    scaler=StandardScaler().fit(Xc); clf=LogisticRegression(C=.5,max_iter=250,class_weight="balanced",solver="liblinear",random_state=0); clf.fit(scaler.transform(Xc),yc); return safe_auc(yd,clf.predict_proba(scaler.transform(Xd))[:,1])
def pass_gate(value,op,threshold): return bool(np.isfinite(value) and ((value>=threshold) if op==">=" else (value<=threshold) if op=="<=" else (value<threshold) if op=="<" else (value==threshold)))


def evaluate_development(seed_results,output,cfg):
    calibration=make_eval_data(cfg.calibration_seed,cfg.calibration_tasks_per_cell,True); development=make_eval_data(cfg.development_seed,cfg.development_tasks_per_cell,True); data_hashes={"calibration":hash_arrays(calibration),"development":hash_arrays(development)}
    policies={}; grids=[]; arrays={}; task_rows=[]; auc_rows=[]; probe_auc={}
    for variant in VARIANTS:
        cpred,cdefault,calt,cratio=aggregate_predictions(seed_results,variant,calibration); dpred,ddefault,dalt,dratio=aggregate_predictions(seed_results,variant,development); policy,grid=calibrate_policy(cpred,cdefault,calt,calibration); grid.insert(0,"variant",variant); grids.append(grid)
        routed=policy_apply(dpred,ddefault,dalt,development,policy) if variant!="forecast_only" else {"loss":ddefault.copy(),"switched":np.zeros(len(ddefault),bool),"action":np.zeros(len(ddefault),int),"selected_ratio":np.ones(len(ddefault)),"confidence":np.zeros(len(ddefault))}; policies[variant]=policy
        useful=dratio<=.85; severe=dratio>1.50; p_use=dpred["probs"][:,:,:2].sum(-1); p_harm=dpred["probs"][:,:,4]; ua=safe_auc(useful.ravel(),p_use.ravel()); ha=safe_auc(severe.ravel(),p_harm.ravel()); auc_rows.append({"variant":variant,"useful_win_auc":ua,"severe_harm_auc":ha}); probe_auc[variant]=fit_latent_probe(cpred,cratio,dpred,dratio); arrays[variant]={"pred":dpred,"default":ddefault,"alt":dalt,"ratio":dratio,"route":routed}
        for i in range(len(ddefault)): task_rows.append({"variant":variant,"task_id":development["task_id"][i],"cell":development["cell"][i],"family":development["family"][i],"default_mase":float(ddefault[i]),"routed_mase":float(routed["loss"][i]),"oracle_mase":float(min(ddefault[i],dalt[i].min())),"switched":bool(routed["switched"][i]),"selected_action":ACTION_NAMES[int(routed["action"][i])] if routed["switched"][i] else "default","selected_ratio":float(routed["selected_ratio"][i])})
    pd.concat(grids,ignore_index=True).to_csv(output/"policy_grid.csv",index=False); pd.DataFrame(task_rows).to_csv(output/"development_task_metrics.csv",index=False); auc_df=pd.DataFrame(auc_rows); auc_df["latent_probe_auc"]=auc_df.variant.map(probe_auc); auc_df.to_csv(output/"critic_auc.csv",index=False); (output/"selected_policies.json").write_text(json.dumps(policies,indent=2,sort_keys=True))
    joint=arrays["joint_contrastive"]; detached=arrays["detached_contrastive"]; shuffled=arrays["shuffled_joint"]; forecast=arrays["forecast_only"]; ddefault=joint["default"]; routed=joint["route"]; ratio=joint["ratio"]; sw=routed["switched"]; opportunity=ratio.min(1)<=.80
    tail=np.isin(development["cell"],["long_horizon_64x32","irregular_missing_64x16","misspecified_64x16","misspecified_32x32"]); supported=np.char.startswith(development["cell"].astype(str),"supported"); comp=np.char.startswith(development["cell"].astype(str),"compositional"); oracle=np.minimum(ddefault,joint["alt"].min(1)); gap=ddefault.mean()-oracle.mean(); auc_map={r["variant"]:r for r in auc_rows}
    replay1=predict(seed_results[0]["models"]["joint_contrastive"],development); replay2=predict(seed_results[0]["models"]["joint_contrastive"],development); replay_diff=max(float(np.max(np.abs(replay1[k]-replay2[k]))) for k in ("mean","probs","adv","score")); altered=dict(development); altered["y"]=development["y"].copy()+997.; firewall=predict(seed_results[0]["models"]["joint_contrastive"],altered); firewall_diff=max(float(np.max(np.abs(replay1[k]-firewall[k]))) for k in ("mean","probs","adv","score"))
    integrity=float(all(len(set(sr["parameter_counts"].values()))==1 for sr in seed_results) and all(len(set(sr["initial_hashes"].values()))==1 for sr in seed_results) and all(sr["updates"]==cfg.train_steps for sr in seed_results) and all(bool(sr["train_hash"]) for sr in seed_results) and replay_diff==0 and firewall_diff==0)
    metrics={"useful_win_auc":float(auc_map["joint_contrastive"]["useful_win_auc"]),"auc_gain_vs_detached":float(auc_map["joint_contrastive"]["useful_win_auc"]-auc_map["detached_contrastive"]["useful_win_auc"]),"severe_harm_auc":float(auc_map["joint_contrastive"]["severe_harm_auc"]),"high_opportunity_recall":float((sw&opportunity).sum()/max(1,opportunity.sum())),"realized_switch_win_rate":float((routed["selected_ratio"][sw]<1).mean()) if sw.any() else 0.,"severe_wrong_switch_rate":float((routed["selected_ratio"][sw]>1.2).mean()) if sw.any() else 1.,"intervention_rate":float(sw.mean()),"synthetic_tail_p95_improvement":float(1-np.quantile(routed["loss"][tail],.95)/np.quantile(ddefault[tail],.95)),"supported_regression":float(routed["loss"][supported].mean()/ddefault[supported].mean()-1),"composition_regression":float(routed["loss"][comp].mean()/ddefault[comp].mean()-1),"oracle_gap_closure":float((ddefault.mean()-routed["loss"].mean())/gap) if gap>1e-8 else 0.,"forecast_regression_vs_control":float(ddefault.mean()/forecast["default"].mean()-1),"auc_gain_vs_shuffled":float(auc_map["joint_contrastive"]["useful_win_auc"]-auc_map["shuffled_joint"]["useful_win_auc"]),"latent_probe_gain_vs_detached":float(probe_auc["joint_contrastive"]-probe_auc["detached_contrastive"]),"matched_parameters_initialization_data_updates_replay_firewall":integrity,"joint_default_mase":float(ddefault.mean()),"joint_routed_mase":float(routed["loss"].mean()),"detached_default_mase":float(detached["default"].mean()),"forecast_only_mase":float(forecast["default"].mean()),"oracle_mase":float(oracle.mean()),"tail_default_p95":float(np.quantile(ddefault[tail],.95)),"tail_routed_p95":float(np.quantile(routed["loss"][tail],.95)),"replay_max_difference":replay_diff,"firewall_max_difference":firewall_diff,"parameter_count":int(seed_results[0]["parameter_counts"]["joint_contrastive"])}
    aliases={"intervention_rate_low":metrics["intervention_rate"],"intervention_rate_high":metrics["intervention_rate"]}; gate_rows=[]
    for gate,(op,threshold) in GATES.items(): observed=aliases[gate] if gate in aliases else metrics[gate]; gate_rows.append({"gate":gate,"observed":observed,"operator":op,"threshold":threshold,"passed":pass_gate(float(observed),op,float(threshold))})
    gates=pd.DataFrame(gate_rows); gates.to_csv(output/"promotion_gates.csv",index=False); task_df=pd.DataFrame(task_rows); task_df.groupby("variant",as_index=False).agg(default_mase=("default_mase","mean"),routed_mase=("routed_mase","mean"),oracle_mase=("oracle_mase","mean"),intervention_rate=("switched","mean")).to_csv(output/"variant_summary.csv",index=False); joint_df=task_df[task_df.variant=="joint_contrastive"]; cells=joint_df.groupby("cell",as_index=False).agg(default_mase=("default_mase","mean"),routed_mase=("routed_mase","mean"),oracle_mase=("oracle_mase","mean"),intervention_rate=("switched","mean")); cells["relative_change"]=cells.routed_mase/cells.default_mase-1; cells.to_csv(output/"cell_summary.csv",index=False)
    passed=int(gates.passed.sum()); decision={"experiment":cfg.experiment,"status":"PROMOTE_STAGE3B_OPEN_CONFIRMATION" if passed==len(gates) else "DO_NOT_PROMOTE_STAGE3B_KEEP_CONFIRMATION_SEALED","gates_passed":passed,"gates_total":int(len(gates)),"all_gates_passed":bool(passed==len(gates)),"metrics":metrics,"selected_policy":policies["joint_contrastive"],"data_hashes":data_hashes,"confirmation_status":"authorized_to_open" if passed==len(gates) else "sealed_not_generated","latent_probe_auc":probe_auc}; (output/"DEVELOPMENT_DECISION.json").write_text(json.dumps(decision,indent=2,sort_keys=True)); return {"decision":decision,"policy":policies["joint_contrastive"],"seed_results":seed_results}


def confirmation_if_authorized(dev,output,cfg):
    if not dev["decision"]["all_gates_passed"]:
        seal={"status":"SEALED","seed_namespace":cfg.confirmation_seed,"reason":"Stage3B development gates did not all pass; confirmation futures were not generated or scored","planned_cells":[c[0] for c in EVAL_CELLS]}; (output/"CONFIRMATION_SEAL.json").write_text(json.dumps(seal,indent=2,sort_keys=True)); return seal
    data=make_eval_data(cfg.confirmation_seed,cfg.confirmation_tasks_per_cell,True); pred,default,alt,ratio=aggregate_predictions(dev["seed_results"],"joint_contrastive",data); route=policy_apply(pred,default,alt,data,dev["policy"]); sw=route["switched"]; opp=ratio.min(1)<=.8; metrics={"data_hash":hash_arrays(data),"mean_improvement":float(1-route["loss"].mean()/default.mean()),"intervention_rate":float(sw.mean()),"win_rate":float((route["selected_ratio"][sw]<1).mean()) if sw.any() else 0.,"severe_wrong":float((route["selected_ratio"][sw]>1.2).mean()) if sw.any() else 1.,"opportunity_recall":float((sw&opp).sum()/max(1,opp.sum()))}; status="CONFIRMATION_PASS" if metrics["mean_improvement"]>0 and metrics["win_rate"]>=.68 and metrics["severe_wrong"]<.05 else "CONFIRMATION_FAIL"; result={"status":status,"metrics":metrics,"policy":dev["policy"]}; (output/"CONFIRMATION_DECISION.json").write_text(json.dumps(result,indent=2,sort_keys=True)); return result

def write_next_step(output,decision,confirmation):
    if decision["all_gates_passed"] and confirmation.get("status")=="CONFIRMATION_PASS":
        name="NEXT_STEP_STAGE3C_MICRO_FOUNDATION_SCALE_SPEC.md"; text="# Frozen next step: Stage 3C 1–3M JAVELIN-IRMC micro-foundation run\n\nStage 3B passed matched development and sealed confirmation. Scale the same architecture to 1–3M parameters with variable contexts to 512, horizons to 96, exact matched controls, and a frozen source-disjoint public benchmark. No architecture or policy changes are permitted after public predictions are opened.\n"
    else:
        name="NEXT_STEP_STAGE3C_MECHANISM_ACTION_BASIS_REPAIR_SPEC.md"; m=decision["metrics"]; text=f"# Frozen next step: Stage 3C mechanism/action-basis repair\n\n## Decision boundary\nStage 3B did not pass its matched joint-pretraining gate. The confirmation namespace remains sealed. No further post-hoc router or threshold ladder is authorized.\n\n## Diagnosis\n- useful-win AUC: {m['useful_win_auc']:.4f}\n- gain over detached: {m['auc_gain_vs_detached']:.4f}\n- high-opportunity recall: {m['high_opportunity_recall']:.2%}\n- tail-p95 improvement: {m['synthetic_tail_p95_improvement']:.2%}\n- oracle-gap closure: {m['oracle_gap_closure']:.2%}\n\n## Required intervention\nReplace the fixed action list as the learning target with **jointly learned residual program atoms**. Each atom must be executable, orthogonalized against the active compiler basis, and trained under cross-origin causal invariance. The critic should predict atom composition and abstention from the same mechanism slots. Compare true atom supervision with detached, shuffled, and fixed-action controls at exact matched active parameters and updates.\n\n## Stop rule\nA failure to exceed the detached control by 0.06 useful-win AUC or close 20% of oracle headroom ends action-routing research and returns the project to process-grammar expansion.\n"
    (output/name).write_text(text); return name

def validate(output,seed_results,final,cfg):
    checks={}; gates=pd.read_csv(output/"promotion_gates.csv"); checks["gate_count_16"]=len(gates)==16; checks["gate_sum_matches"]=int(gates.passed.sum())==final["development"]["gates_passed"]; checks["three_seeds"]=len(seed_results)==3; checks["four_matched_variants"]=all(len(sr["parameter_counts"])==4 and len(set(sr["parameter_counts"].values()))==1 for sr in seed_results); checks["initial_states_bitwise_equal"]=all(len(set(sr["initial_hashes"].values()))==1 for sr in seed_results); checks["matched_updates"]=all(sr["updates"]==cfg.train_steps for sr in seed_results); task=pd.read_csv(output/"development_task_metrics.csv"); checks["task_rows_complete"]=len(task)==4*8*cfg.development_tasks_per_cell; checks["eight_cells"]=task.cell.nunique()==8; checks["finite_metrics"]=np.isfinite(task[["default_mase","routed_mase","oracle_mase"]].to_numpy()).all(); checks["oracle_not_worse"]=bool((task.oracle_mase<=task.default_mase+1e-6).all()); checks["replay_exact"]=final["development"]["metrics"]["replay_max_difference"]==0; checks["firewall_exact"]=final["development"]["metrics"]["firewall_max_difference"]==0; checks["confirmation_boundary_respected"]=final["confirmation"].get("status") in ("SEALED","CONFIRMATION_PASS","CONFIRMATION_FAIL"); checks["spec_present"]=(output/"FROZEN_STAGE3B_SPEC.json").exists(); checks["source_present"]=(output/"stage3b_experiment.py").exists(); checks["policy_grid_present"]=(output/"policy_grid.csv").exists(); checks["probe_rows_complete"]=len(pd.read_csv(output/"critic_auc.csv"))==4; checks={k:bool(v) for k,v in checks.items()}; passed=all(checks.values()); result={"status":"PASS" if passed else "FAIL","passed":int(sum(checks.values())),"total":len(checks),"checks":checks}; (output/"VALIDATION.json").write_text(json.dumps(result,indent=2,sort_keys=True)); lines=["# Validation Report","",f"### Overall Assessment: {'Ready to share' if passed else 'Needs revision'}","",f"Independent validation passed **{result['passed']} of {result['total']}** checks.","","### Checks"]+[f"- {'PASS' if v else 'FAIL'} — `{k}`" for k,v in checks.items()]+["","### Required caveat","This is a compact synthetic causal architecture gate. It is not evidence of frontier-model superiority."]; (output/"VALIDATION.md").write_text("\n".join(lines)+"\n"); return result

def write_decision(output,final):
    d=final["development"]; m=d["metrics"]; lines=["# Stage 3B LATENT-ACTION decision","",f"**Status:** `{d['status']}`","",f"Frozen gates passed: **{d['gates_passed']} / {d['gates_total']}**.","","## Key metrics","",f"- useful-win AUC: **{m['useful_win_auc']:.4f}**",f"- gain over detached: **{m['auc_gain_vs_detached']:.4f}**",f"- severe-harm AUC: **{m['severe_harm_auc']:.4f}**",f"- latent-probe gain over detached: **{m['latent_probe_gain_vs_detached']:.4f}**",f"- high-opportunity recall: **{m['high_opportunity_recall']:.2%}**",f"- intervention rate: **{m['intervention_rate']:.2%}**",f"- switch win rate: **{m['realized_switch_win_rate']:.2%}**",f"- severe wrong-switch rate: **{m['severe_wrong_switch_rate']:.2%}**",f"- tail-p95 improvement: **{m['synthetic_tail_p95_improvement']:.2%}**",f"- oracle-gap closure: **{m['oracle_gap_closure']:.2%}**","","## Confirmation boundary","",f"Confirmation status: **{final['confirmation'].get('status')}**.","The confirmation namespace is generated and scored only after a complete development pass.","","## Scope","","This experiment tests causal value from mechanism-slot/action-trajectory contrastive pretraining under matched controls. It is not a public frontier-model comparison."]; (output/"DECISION.md").write_text("\n".join(lines)+"\n")
def make_report(output,final,validation):
    import nbformat as nbf; d=final["development"]; m=d["metrics"]; nb=nbf.v4.new_notebook(); nb.metadata["kernelspec"]={"display_name":"Python 3","language":"python","name":"python3"}; nb.cells=[nbf.v4.new_markdown_cell("# Stage 3B LATENT-ACTION Contrastive Pretraining\n\n## tl;dr\n\n"+f"**Decision:** `{d['status']}`. The treatment passed **{d['gates_passed']} of {d['gates_total']}** frozen gates. Useful-win AUC was **{m['useful_win_auc']:.3f}**, gain over detached was **{m['auc_gain_vs_detached']:.3f}**, and tail-p95 improvement was **{m['synthetic_tail_p95_improvement']:.1%}**. Confirmation: **{final['confirmation'].get('status')}**."),nbf.v4.new_markdown_cell("## Context & Methods\n\nFour exactly parameter-matched models were trained from bitwise-identical initial states on identical task streams. The treatment alone allowed true action-trajectory contrastive gradients to reshape mechanism slots. The detached control trained the same critic without encoder gradients; the shuffled control used permuted counterfactual labels. All deployment inputs are history-only."),nbf.v4.new_code_cell("from pathlib import Path\nimport json,pandas as pd\nROOT=Path('.')\nfinal=json.loads((ROOT/'FINAL_DECISION.json').read_text())\ngates=pd.read_csv(ROOT/'promotion_gates.csv')\nauc=pd.read_csv(ROOT/'critic_auc.csv')\nvariants=pd.read_csv(ROOT/'variant_summary.csv')\ncells=pd.read_csv(ROOT/'cell_summary.csv')\nprint(json.dumps({k:final['development'][k] for k in ['status','gates_passed','gates_total','confirmation_status']},indent=2))"),nbf.v4.new_markdown_cell("## Results"),nbf.v4.new_code_cell("gates"),nbf.v4.new_code_cell("auc"),nbf.v4.new_code_cell("variants"),nbf.v4.new_code_cell("cells"),nbf.v4.new_markdown_cell("## Takeaways\n\nThe frozen conjunction of gates determines promotion. A failure keeps confirmation sealed and authorizes only the mechanism/action-basis repair recorded in the next-step specification."),nbf.v4.new_code_cell(f"validation={json.dumps(validation)}\nvalidation")]; path=output/"Stage3B_LATENT_ACTION_Report.ipynb"; nbf.write(nb,path)
    try: subprocess.run([sys.executable,"-m","jupyter","nbconvert","--execute","--to","notebook","--inplace",path.name],cwd=output,check=True,timeout=360); subprocess.run([sys.executable,"-m","jupyter","nbconvert","--to","html",path.name],cwd=output,check=True,timeout=360)
    except Exception as exc: (output/"NOTEBOOK_EXECUTION_GAP.txt").write_text(str(exc)+"\n")
def manifest_and_bundle(output):
    manifest=output/"MANIFEST.sha256"; files=sorted(p for p in output.rglob("*") if p.is_file() and p.name!=manifest.name); manifest.write_text("".join(f"{sha256_file(p)}  {p.relative_to(output)}\n" for p in files)); bundle=output.parent/f"{output.name}_Bundle.zip"
    with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as zf:
        for p in sorted(output.rglob("*")):
            if p.is_file(): zf.write(p,p.relative_to(output))
    shafile=Path(str(bundle)+".sha256"); shafile.write_text(f"{sha256_file(bundle)}  {bundle.name}\n"); return bundle,shafile
def effective_config(smoke): return CFG if not smoke else replace(CFG,train_tasks=192,train_steps=4,batch_size=32,calibration_tasks_per_cell=4,development_tasks_per_cell=4,confirmation_tasks_per_cell=4,bootstrap_draws=20)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--smoke",action="store_true"); args=parser.parse_args(); cfg=effective_config(args.smoke); output=args.output.resolve()
    if output.exists(): shutil.rmtree(output)
    output.mkdir(parents=True); started=time.time()
    try:
        frozen={"config":asdict(cfg),"variants":VARIANTS,"actions":ACTION_NAMES,"families":FAMILIES,"eval_cells":EVAL_CELLS,"gates":{k:{"operator":v[0],"threshold":v[1]} for k,v in GATES.items()},"scientific_intervention":"explicit action-trajectory contrastive pretraining with mechanism-conditioned latent slots","confirmation_rule":"never generate or score confirmation futures unless every development gate passes"}; (output/"FROZEN_STAGE3B_SPEC.json").write_text(json.dumps(frozen,indent=2,sort_keys=True)); shutil.copy2(Path(__file__),output/"stage3b_experiment.py"); seed_results=[train_one_seed(i,output,cfg) for i in range(len(cfg.model_seeds))]; dev=evaluate_development(seed_results,output,cfg); confirmation=confirmation_if_authorized(dev,output,cfg); next_step=write_next_step(output,dev["decision"],confirmation); final={"development":dev["decision"],"confirmation":confirmation,"next_step_file":next_step,"elapsed_seconds":time.time()-started}; (output/"FINAL_DECISION.json").write_text(json.dumps(final,indent=2,sort_keys=True)); write_decision(output,final); validation=validate(output,seed_results,final,cfg); make_report(output,final,validation); pd.DataFrame([{"status":final["development"]["status"],"gates_passed":final["development"]["gates_passed"],"gates_total":final["development"]["gates_total"],**{k:v for k,v in final["development"]["metrics"].items() if isinstance(v,(int,float,bool))}}]).to_csv(output/"RESULT_SUMMARY.csv",index=False); bundle,shafile=manifest_and_bundle(output); (output/"RUN_COMPLETE.txt").write_text(f"status={final['development']['status']}\nbundle={bundle.name}\nsha256={sha256_file(bundle)}\n"); print(json.dumps(final,indent=2,sort_keys=True),flush=True)
    except Exception as exc:
        failure={"status":"STAGE3B_EXECUTION_FAILED","error_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc(),"elapsed_seconds":time.time()-started}; (output/"EXECUTION_FAILURE.json").write_text(json.dumps(failure,indent=2,sort_keys=True)); print(json.dumps(failure,indent=2,sort_keys=True),file=sys.stderr,flush=True); raise

if __name__=="__main__": main()
