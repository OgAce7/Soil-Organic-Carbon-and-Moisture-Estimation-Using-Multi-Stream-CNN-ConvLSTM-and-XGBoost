"""
plot_soc.py -- quick overview figure of the cleaned WoSIS SOC samples.
Run from project root:  python scripts/plot_soc.py
Reads  data/raw/samples_clean.csv   Saves  data/raw/soc_overview.png
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/samples_clean.csv")
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.5, 1, 1]})

sc = ax[0].scatter(df["lon"], df["lat"], c=df["soc"], s=6, cmap="viridis_r",
                   vmin=0, vmax=np.percentile(df["soc"], 95))
ax[0].set(title="WoSIS topsoil SOC samples (0-30 cm), n = {:,}".format(len(df)),
          xlabel="Longitude", ylabel="Latitude")
ax[0].set_aspect(1 / np.cos(np.deg2rad(df["lat"].mean())))
fig.colorbar(sc, ax=ax[0], label="SOC (g/kg)", shrink=0.85)

ax[1].hist(df["soc"], bins=50, color="#2a7f62", edgecolor="white")
ax[1].axvline(df["soc"].median(), color="k", ls="--", lw=1, label="median {:.1f}".format(df["soc"].median()))
ax[1].set(title="SOC distribution", xlabel="SOC (g/kg)", ylabel="Samples")
ax[1].legend()

top = df["country"].value_counts().head(10)[::-1]
ax[2].barh(top.index, top.values, color="#3b6ea5")
ax[2].set(title="Samples per country (top 10)", xlabel="Samples")

fig.suptitle("Data: ISRIC WoSIS latest (doi:10.5194/essd-16-4735-2024)", fontsize=9, y=0.02)
fig.tight_layout()
fig.savefig("data/raw/soc_overview.png", dpi=160, bbox_inches="tight")
print("saved data/raw/soc_overview.png")