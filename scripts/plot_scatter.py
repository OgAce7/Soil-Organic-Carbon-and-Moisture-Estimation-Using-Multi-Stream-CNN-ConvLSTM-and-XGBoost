"""
plot_scatter.py -- predicted vs observed SOC for the trained CNN-LSTM.

Run from the project root, after train.py has saved a model:
    python scripts/plot_scatter.py

Reads : the data files and model path defined in config.py
Writes: results/scatter_pred_vs_obs.png   (test scatter + residual histogram)
        results/predictions_test.csv      (lon, lat, observed, predicted, residual for the test set)
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

import config as cfg
import pred
import utils

if cfg.model_name != "CNN-LSTM_evi_lsp":
    sys.exit("This script is written for model_name = 'CNN-LSTM_evi_lsp' (got '{}').".format(cfg.model_name))

x_cnn, _, _, x_lstm, y = utils.generate_xy()
train_idx = utils.load_pickle(cfg.f_train_index)
test_idx = utils.load_pickle(cfg.f_test_index)

model = pred.get_model(model_name=cfg.model_name, model_save_pth=cfg.model_save_pth)
if cfg.device == "cuda":
    model = model.cuda()


def run(idx):
    p = np.array(pred.predict(model=model, x_data_cnn=x_cnn[idx], x_data_lstm=x_lstm[idx])).ravel()
    t = np.asarray(y)[idx].ravel()
    if hasattr(utils, "inv_target"):      # model trained on log(SOC): convert both back to g/kg
        return utils.inv_target(t), utils.inv_target(p)
    return t, p


y_te, p_te = run(test_idx)
y_tr, p_tr = run(train_idx)


def scores(a, b):
    return (np.sqrt(metrics.mean_squared_error(a, b)), metrics.mean_absolute_error(a, b), metrics.r2_score(a, b))


rm_te, ma_te, r2_te = scores(y_te, p_te)
rm_tr, ma_tr, r2_tr = scores(y_tr, p_tr)
print("TEST : RMSE {:.2f}  MAE {:.2f}  R2 {:.3f}".format(rm_te, ma_te, r2_te))
print("TRAIN: RMSE {:.2f}  MAE {:.2f}  R2 {:.3f}".format(rm_tr, ma_tr, r2_tr))

os.makedirs("results", exist_ok=True)
samples = pd.read_csv(cfg.f_df_samples).iloc[test_idx].reset_index(drop=True)
pd.DataFrame({"lon": samples["lon"], "lat": samples["lat"], "observed": y_te,
              "predicted": p_te, "residual": p_te - y_te}).to_csv("results/predictions_test.csv", index=False)

lim = float(np.percentile(np.concatenate([y_te, p_te]), 99.5)) * 1.05
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})

ax[0].scatter(y_te, p_te, s=12, alpha=0.55, color="#2a7f62", edgecolor="none")
ax[0].plot([0, lim], [0, lim], "k--", lw=1, label="1:1 line")
slope, intercept = np.polyfit(y_te, p_te, 1)
ax[0].plot([0, lim], [intercept, intercept + slope * lim], color="#c0392b", lw=1.5, label="fitted line")
ax[0].set(xlim=(0, lim), ylim=(0, lim), xlabel="Observed SOC (g/kg)", ylabel="Predicted SOC (g/kg)",
          title="Test set: predicted vs observed (n = {})".format(len(y_te)))
ax[0].set_aspect("equal")
ax[0].text(0.04, 0.96, "R$^2$ = {:.3f}\nRMSE = {:.2f} g/kg\nMAE = {:.2f} g/kg".format(r2_te, rm_te, ma_te),
           transform=ax[0].transAxes, va="top", bbox=dict(boxstyle="round", fc="white", ec="0.7"))
ax[0].legend(loc="lower right")

res = p_te - y_te
ax[1].hist(res, bins=40, color="#3b6ea5", edgecolor="white")
ax[1].axvline(0, color="k", lw=1)
ax[1].axvline(res.mean(), color="#c0392b", ls="--", lw=1.2, label="mean bias {:+.2f}".format(res.mean()))
ax[1].set(xlabel="Predicted - observed (g/kg)", ylabel="Samples", title="Test residuals")
ax[1].legend()

fig.suptitle("CNN-LSTM SOC model  |  train R$^2$ = {:.3f}, test R$^2$ = {:.3f}".format(r2_tr, r2_te), fontsize=10)
fig.tight_layout()
fig.savefig("results/scatter_pred_vs_obs.png", dpi=160, bbox_inches="tight")
print("saved results/scatter_pred_vs_obs.png and results/predictions_test.csv")