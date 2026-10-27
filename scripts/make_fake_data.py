"""
make_fake_data.py -- SIMULATED data for testing the CNN-LSTM SOC pipeline.

!! This data is synthetic. Results (R2, RMSE) say nothing about real soil. !!

Run from the project root (the folder containing config.py):
    python scripts/make_fake_data.py

Writes (paths come from config.py, so they always match what train.py reads):
    data/samples_data.csv            columns: lon, lat, soc
    data/samples_window_common.pkl   (N, 10, 7, 7)   CNN input   (static layers, 7x7 pixel window)
    data/samples_ts_evi.pkl          (N, T, 1)       LSTM input  (EVI time series)
    data/samples_ts_lsp.pkl          (N, T, 11)      LSTM input  (phenology features)
    data/train_test_idx/train_1.pkl, test_1.pkl      (80/20 split of row indices)
Also creates model/ and log/ folders.

SOC is built from the simulated inputs (plus noise) so the model has real
signal to learn instead of pure noise. Change the constants below to resize.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# make `import config` work when run as scripts/make_fake_data.py
sys.path.insert(0, os.getcwd())
import config as cfg
import utils

N = 1000          # number of samples
WIN = 7           # window size in pixels; MUST be 6-8 for models.py (Linear(64, 16))
T = 24            # time steps (e.g. 24 = two 12-month years, or 24 x 16-day composites)
N_STATIC = cfg.num_channels        # 10
N_EVI = cfg.lstm_input_size_evi    # 1
N_LSP = cfg.lstm_input_size_lsp    # 11
SEED = cfg.rand_seed

rng = np.random.default_rng(SEED)


def standardize(a, axis):
    """zero mean / unit variance along the given axes (the real pipeline should do this too)."""
    return (a - a.mean(axis=axis, keepdims=True)) / (a.std(axis=axis, keepdims=True) + 1e-8)


# ---- sample locations: random points inside a rough Europe bounding box ----
lon = rng.uniform(-10, 30, N)
lat = rng.uniform(36, 60, N)

# ---- latent site conditions that drive everything else ----
elev = rng.normal(0, 1, N)       # elevation-like
temp = -0.7 * elev + rng.normal(0, 0.7, N)   # colder when higher
precip = 0.4 * elev + rng.normal(0, 0.9, N)
wet = rng.normal(0, 1, N)        # topographic wetness-like
site = np.stack([elev, temp, precip, wet], axis=1)

# ---- CNN input: (N, 10, WIN, WIN) ----
# each channel = site value + gentle spatial gradient + pixel noise
yy, xx = np.meshgrid(np.linspace(-1, 1, WIN), np.linspace(-1, 1, WIN), indexing="ij")
x_cnn = np.zeros((N, N_STATIC, WIN, WIN), dtype=np.float32)
for c in range(N_STATIC):
    base = site[:, c % 4] * rng.uniform(0.6, 1.0) + rng.normal(0, 0.3, N)
    gx, gy = rng.normal(0, 0.3, (2, N))
    field = base[:, None, None] + gx[:, None, None] * xx + gy[:, None, None] * yy
    field += rng.normal(0, 0.15, (N, WIN, WIN))
    x_cnn[:, c] = field
x_cnn = standardize(x_cnn, axis=(0, 2, 3)).astype(np.float32)   # per-channel

# ---- EVI time series: (N, T, 1), seasonal curve whose amplitude/level depend on climate ----
t = np.arange(T)
level = 0.35 + 0.08 * precip - 0.04 * temp + rng.normal(0, 0.03, N)
amp = 0.20 + 0.05 * precip + rng.normal(0, 0.02, N)
phase = rng.normal(0, 0.4, N)
season = np.sin(2 * np.pi * t[None, :] / 12.0 + phase[:, None])
evi = level[:, None] + amp[:, None] * season + rng.normal(0, 0.02, (N, T))
x_evi = standardize(evi[:, :, None], axis=(0, 1)).astype(np.float32)

# ---- phenology: (N, T, 11), different noisy views of season timing/amplitude ----
lsp = np.zeros((N, T, N_LSP), dtype=np.float32)
for k in range(N_LSP):
    w = rng.uniform(0.3, 1.0)
    lsp[:, :, k] = (
        w * amp[:, None] * np.cos(2 * np.pi * t[None, :] / 12.0 + phase[:, None] + k * 0.3)
        + 0.5 * level[:, None]
        + rng.normal(0, 0.05, (N, T))
    )
x_lsp = standardize(lsp, axis=(0, 1)).astype(np.float32)

# ---- target: SOC in g/kg, built from inputs + noise (so it is learnable) ----
cnn_mean = x_cnn.mean(axis=(2, 3))            # (N, 10) window means
soc = (
    20.0
    + 5.0 * cnn_mean[:, 3]       # "wetness" channel
    - 4.0 * cnn_mean[:, 1]       # "temperature" channel
    + 3.0 * cnn_mean[:, 2]       # "precipitation" channel
    + 6.0 * x_evi.mean(axis=(1, 2))
    + 3.0 * x_evi.std(axis=(1, 2))
    + rng.normal(0, 3.0, N)
)
soc = np.clip(soc, 2.0, None).astype(np.float32)

# ---- write everything ----
os.makedirs(cfg.data_dir, exist_ok=True)
os.makedirs(os.path.dirname(cfg.f_train_index), exist_ok=True)
os.makedirs(os.path.dirname(cfg.model_save_pth) or ".", exist_ok=True)
os.makedirs(cfg.log_dir, exist_ok=True)

pd.DataFrame({"lon": lon, "lat": lat, cfg.target_var_name: soc}).to_csv(cfg.f_df_samples, index=False)
utils.save_pickle(cfg.f_data_DL_common, x_cnn)
utils.save_pickle(cfg.f_data_DL_evi, x_evi)
utils.save_pickle(cfg.f_data_DL_lsp, x_lsp)

idx = np.arange(N)
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED)
utils.save_pickle(cfg.f_train_index, train_idx)
utils.save_pickle(cfg.f_test_index, test_idx)

print("Wrote SIMULATED data:")
print("  samples_data.csv           :", (N, 3))
print("  samples_window_common.pkl  :", x_cnn.shape)
print("  samples_ts_evi.pkl         :", x_evi.shape)
print("  samples_ts_lsp.pkl         :", x_lsp.shape)
print("  train / test               :", len(train_idx), "/", len(test_idx))
print("  SOC (g/kg) mean/std/min/max: {:.1f} / {:.1f} / {:.1f} / {:.1f}".format(soc.mean(), soc.std(), soc.min(), soc.max()))