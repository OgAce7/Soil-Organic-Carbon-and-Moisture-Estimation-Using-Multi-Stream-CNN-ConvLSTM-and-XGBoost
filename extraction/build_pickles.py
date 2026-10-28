"""
build_pickles.py -- pull model inputs from Google Earth Engine for the WoSIS SOC points
and write the files train.py expects.

Run from the project root, venv active:
    python extraction/build_pickles.py --project YOUR_GCP_PROJECT_ID --n 300     # small test first
    python extraction/build_pickles.py --project YOUR_GCP_PROJECT_ID --n 2000    # bigger run
    python extraction/build_pickles.py --project YOUR_GCP_PROJECT_ID --n 0       # all points (slow)

Inputs : data/raw/samples_clean.csv   (from clean_soc.py)
Outputs: data/samples_data.csv                lon, lat, soc          (row i = sample i everywhere)
         data/samples_window_common.pkl       (N, 10, 7, 7)  static layers, 7x7 pixel window
         data/samples_ts_evi.pkl              (N, 60, 1)     monthly EVI, 2003-2007
         data/samples_ts_lsp.pkl              (N, 60, 11)    yearly phenology repeated over its 12 months
         data/train_test_idx/train_1.pkl, test_1.pkl
Existing data/ files (e.g. the SIMULATED ones) are moved to data/fake_backup/ first.

Grid: everything is resampled to the MODIS 250 m grid, so a 7x7 window is ~1.6 km across.
Static layers (10): elevation, slope, aspect_sin, aspect_cos, hillshade (SRTM 30 m, averaged to 250 m)
                    + WorldClim v1 bio01, bio04, bio05, bio06, bio12 (1 km).
Phenology (11): Greenup, MidGreenup, Peak, Maturity, Senescence, MidGreendown, Dormancy (as day-of-year),
                EVI_Minimum, EVI_Amplitude, EVI_Area, NumCycles   (first cycle of each year).
Limitation: results are cached per chunk in data/raw/gee_cache/, so an interrupted run can be resumed.
"""
import argparse
import glob
import math
import os
import pickle
import shutil
import time

import sys

import numpy as np
import pandas as pd

# make `import config` work when run as extraction/build_pickles.py from the project root
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
import utils

# ------------------------------------------------------------------ settings
START_YEAR, N_YEARS = 2003, 5
YEARS = list(range(START_YEAR, START_YEAR + N_YEARS))
T = N_YEARS * 12
HALF = 3                                   # kernel radius -> 7x7 window
CHUNK = 150                                # points per GEE request
CACHE = "data/raw/gee_cache"
MISSING = -9999                            # placeholder for masked pixels (EE drops points with any masked band)

STATIC_BANDS = ["elevation", "slope", "aspect_sin", "aspect_cos", "hillshade",
                "bio01", "bio04", "bio05", "bio06", "bio12"]
PH_DATE = ["Greenup_1", "MidGreenup_1", "Peak_1", "Maturity_1", "Senescence_1", "MidGreendown_1", "Dormancy_1"]
PH_OTHER = ["EVI_Minimum_1", "EVI_Amplitude_1", "EVI_Area_1", "NumCycles"]
PH_BANDS = PH_DATE + PH_OTHER              # 11 features


# ------------------------------------------------------------------ GEE image builders
def build_images(ee):
    mod = ee.ImageCollection("MODIS/061/MOD13Q1")
    proj = mod.first().select("EVI").projection()

    ter = ee.Terrain.products(ee.Image("USGS/SRTMGL1_003"))
    asp = ter.select("aspect").multiply(math.pi / 180)
    terrain = ee.Image.cat([
        ter.select("elevation"), ter.select("slope"),
        asp.sin().rename("aspect_sin"), asp.cos().rename("aspect_cos"),
        ter.select("hillshade"),
    ]).reduceResolution(ee.Reducer.mean(), maxPixels=1024).reproject(proj)
    clim = ee.Image("WORLDCLIM/V1/BIO").select(["bio01", "bio04", "bio05", "bio06", "bio12"]).reproject(proj)
    static = ee.Image.cat([terrain, clim]).select(STATIC_BANDS).neighborhoodToArray(ee.Kernel.square(HALF))

    def good(i):
        return i.select("EVI").updateMask(i.select("SummaryQA").lte(1))

    evi_bands = []
    for y in YEARS:
        for m in range(1, 13):
            s = ee.Date.fromYMD(y, m, 1)
            evi_bands.append(mod.filterDate(s, s.advance(1, "month")).map(good).mean().rename("evi_%d" % len(evi_bands)))
    evi = ee.Image.cat(evi_bands).unmask(MISSING)

    ph_col = ee.ImageCollection("MODIS/061/MCD12Q2")
    ph_bands = []
    for k, y in enumerate(YEARS):
        img = ph_col.filterDate("%d-01-01" % y, "%d-01-01" % (y + 1)).first()
        base = ee.Date.fromYMD(y, 1, 1).millis().divide(86400000)    # days since 1970-01-01
        dates = img.select(PH_DATE).subtract(base)                    # -> day-of-year (approx., can be <0 or >365)
        other = img.select(PH_OTHER)
        ph_bands.append(ee.Image.cat([dates, other]).rename(["p%d_%s" % (k, b) for b in PH_BANDS]))
    pheno = ee.Image.cat(ph_bands).unmask(MISSING)
    return proj, static, evi, pheno


# ------------------------------------------------------------------ parsing GEE results
def parse_static(feats):
    out = {}
    for f in feats:
        p = f["properties"]
        try:
            arr = np.stack([np.array(p[b], dtype=float) for b in STATIC_BANDS])   # (10, 7, 7)
        except (KeyError, TypeError, ValueError):
            continue
        if arr.shape == (len(STATIC_BANDS), 2 * HALF + 1, 2 * HALF + 1):
            out[int(p["pid"])] = arr
    return out


def parse_evi(feats):
    out = {}
    for f in feats:
        p = f["properties"]
        vals = [p.get("evi_%d" % t) for t in range(T)]
        out[int(p["pid"])] = np.array([np.nan if (v is None or v == MISSING) else v * 0.0001 for v in vals], dtype=float)
    return out


def parse_pheno(feats):
    out = {}
    for f in feats:
        p = f["properties"]
        arr = np.full((N_YEARS, len(PH_BANDS)), np.nan)
        for k in range(N_YEARS):
            for j, b in enumerate(PH_BANDS):
                v = p.get("p%d_%s" % (k, b))
                if v is not None and v != MISSING:
                    arr[k, j] = v
        out[int(p["pid"])] = arr
    return out


def fetch(call, tries=4):
    for i in range(tries):
        try:
            return call()
        except Exception as e:  # GEE hiccups / quota
            print("   retry {}/{} after error: {}".format(i + 1, tries, str(e)[:120]))
            time.sleep(5 * (i + 1))
    raise RuntimeError("GEE request kept failing")


# ------------------------------------------------------------------ assembly
def assemble(df, S, E, P):
    keep = [pid for pid in df["pid"] if pid in S]
    print("points with a full valid 7x7 static window: {} of {}".format(len(keep), len(df)))
    x_cnn = np.stack([S[p] for p in keep]).astype(np.float32)
    ok = ~np.isnan(x_cnn).any(axis=(1, 2, 3))

    evi = np.stack([E.get(p, np.full(T, np.nan)) for p in keep])
    ph = np.stack([P.get(p, np.full((N_YEARS, len(PH_BANDS)), np.nan)) for p in keep])

    # EVI: linear interpolation over gaps; drop samples with no EVI at all
    idx = np.arange(T)
    for i in range(len(evi)):
        v = ~np.isnan(evi[i])
        if v.sum() == 0:
            ok[i] = False
        elif v.sum() < T:
            evi[i] = np.interp(idx, idx[v], evi[i][v])
    # phenology: fill gaps with the per-feature median
    med = np.nanmedian(ph.reshape(-1, ph.shape[-1]), axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    for j in range(ph.shape[-1]):
        m = np.isnan(ph[:, :, j])
        ph[:, :, j][m] = med[j]

    keep = np.array(keep)[ok]
    x_cnn, evi, ph = x_cnn[ok], evi[ok], ph[ok]
    print("final samples:", len(keep))

    lsp = np.repeat(ph, 12, axis=1)                  # yearly value held over its 12 monthly steps -> (N, T, 11)
    x_evi = evi[:, :, None]

    def z(a, axes):
        return ((a - a.mean(axis=axes, keepdims=True)) / (a.std(axis=axes, keepdims=True) + 1e-8)).astype(np.float32)

    return keep, z(x_cnn, (0, 2, 3)), z(x_evi, (0, 1)), z(lsp, (0, 1))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="your Google Cloud project id")
    ap.add_argument("--n", type=int, default=300, help="number of points (0 = all)")
    ap.add_argument("--seed", type=int, default=cfg.rand_seed)
    a = ap.parse_args()

    import ee
    ee.Initialize(project=a.project)

    df = pd.read_csv("data/raw/samples_clean.csv")
    df = df.dropna(subset=["lon", "lat", "soc"]).reset_index(drop=True)
    if a.n and a.n < len(df):
        df = df.sample(a.n, random_state=a.seed).reset_index(drop=True)
    df["pid"] = np.arange(len(df))
    print("points to extract:", len(df), "| window {}-{}".format(YEARS[0], YEARS[-1]))

    proj, static, evi, pheno = build_images(ee)
    os.makedirs(CACHE, exist_ok=True)
    S, E, P = {}, {}, {}
    for start in range(0, len(df), CHUNK):
        part = df.iloc[start:start + CHUNK]
        cfile = os.path.join(CACHE, "v2_n{}_seed{}_{}.pkl".format(len(df), a.seed, start))
        if os.path.exists(cfile):
            with open(cfile, "rb") as f:
                s, e, p = pickle.load(f)
            print("chunk {} loaded from cache".format(start))
        else:
            print("chunk {}-{} ...".format(start, start + len(part)), flush=True)
            fc = ee.FeatureCollection([
                ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {"pid": int(r.pid)})
                for r in part.itertuples()])
            s = parse_static(fetch(lambda: static.sampleRegions(collection=fc, properties=["pid"], projection=proj, tileScale=4).getInfo())["features"])
            e = parse_evi(fetch(lambda: evi.sampleRegions(collection=fc, properties=["pid"], scale=250, tileScale=4).getInfo())["features"])
            p = parse_pheno(fetch(lambda: pheno.sampleRegions(collection=fc, properties=["pid"], scale=500, tileScale=4).getInfo())["features"])
            with open(cfile, "wb") as f:
                pickle.dump((s, e, p), f)
        S.update(s); E.update(e); P.update(p)

    keep, x_cnn, x_evi, x_lsp = assemble(df, S, E, P)
    out = df.set_index("pid").loc[keep, ["lon", "lat", "soc"]].reset_index(drop=True)

    # back up whatever is in data/ now (e.g. the simulated files), then write the real ones
    old = [f for f in glob.glob("data/*.pkl") + glob.glob("data/*.csv")]
    if old and not os.path.isdir("data/fake_backup"):
        os.makedirs("data/fake_backup")
        for f in old:
            shutil.move(f, os.path.join("data/fake_backup", os.path.basename(f)))
        print("moved old data files to data/fake_backup/")
    else:                                    # backup already exists: just replace the previous real-data run
        for f in old:
            os.remove(f)
    os.makedirs(os.path.dirname(cfg.f_train_index), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.model_save_pth) or ".", exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    from sklearn.model_selection import train_test_split
    out.to_csv(cfg.f_df_samples, index=False)
    utils.save_pickle(cfg.f_data_DL_common, x_cnn)
    utils.save_pickle(cfg.f_data_DL_evi, x_evi)
    utils.save_pickle(cfg.f_data_DL_lsp, x_lsp)
    tr, te = train_test_split(np.arange(len(out)), test_size=0.2, random_state=a.seed)
    utils.save_pickle(cfg.f_train_index, tr)
    utils.save_pickle(cfg.f_test_index, te)

    print("\nWROTE REAL-DATA FILES")
    print("  samples_data.csv          :", out.shape)
    print("  samples_window_common.pkl :", x_cnn.shape)
    print("  samples_ts_evi.pkl        :", x_evi.shape)
    print("  samples_ts_lsp.pkl        :", x_lsp.shape)
    print("  train / test              :", len(tr), "/", len(te))
    print("  SOC mean/std              : {:.1f} / {:.1f}".format(out.soc.mean(), out.soc.std()))


if __name__ == "__main__":
    main()