"""
clean_soc.py -- turn WoSIS layer rows into ONE topsoil SOC value per profile.

Input : data/raw/wosis_orgc_europe.csv   (from get_wosis.py fetch)
Output: data/raw/samples_clean.csv       columns: profile_id, lon, lat, soc, year, country, licence
        (deliberately NOT data/samples_data.csv, so the fake-data files keep working;
         copy it over later when the satellite pickles for these points exist)

Method: for each profile, depth-weighted mean of organic carbon over the 0-30 cm
interval, using only the part of each layer that overlaps 0-30 cm. Profiles whose
layers cover less than MIN_COVER_CM of that interval are dropped.

Run from the project root:  python scripts/clean_soc.py
"""
import numpy as np
import pandas as pd

IN = "data/raw/wosis_orgc_europe.csv"
OUT = "data/raw/samples_clean.csv"

TOP, BOTTOM = 0.0, 30.0   # cm, depth interval to summarise
MIN_COVER_CM = 15.0       # require at least this much of 0-30 cm to be measured
MAX_SOC = 200.0           # g/kg; above this = organic/peat soils, dropped (set None to keep)
MIN_YEAR = None           # e.g. 2000 to keep only profiles sampled in the MODIS era

df = pd.read_csv(IN)
print("raw rows:", len(df), "| profiles:", df["profile_id"].nunique())

# value_avg = WoSIS-averaged value; fall back to 'value' where it is missing
df["oc"] = pd.to_numeric(df["value_avg"], errors="coerce")
df["oc"] = df["oc"].fillna(pd.to_numeric(df["value"], errors="coerce"))
df["upper_depth"] = pd.to_numeric(df["upper_depth"], errors="coerce")
df["lower_depth"] = pd.to_numeric(df["lower_depth"], errors="coerce")
df = df.dropna(subset=["oc", "upper_depth", "lower_depth", "lon", "lat"])
df = df[df["lower_depth"] > df["upper_depth"]]

# thickness of each layer that falls inside 0-30 cm
overlap = (np.minimum(df["lower_depth"], BOTTOM) - np.maximum(df["upper_depth"], TOP)).clip(lower=0)
df = df.assign(w=overlap)
df = df[df["w"] > 0]
df["w_oc"] = df["w"] * df["oc"]

g = df.groupby("profile_id")
out = pd.DataFrame({
    "soc": g["w_oc"].sum() / g["w"].sum(),
    "cover_cm": g["w"].sum(),
    "lon": g["lon"].first(),
    "lat": g["lat"].first(),
    "date": g["date"].first(),
    "country": g["country_name"].first(),
    "licence": g["licence"].first(),
}).reset_index()
print("profiles with any 0-30 cm data:", len(out))

out = out[out["cover_cm"] >= MIN_COVER_CM]
print("after coverage >= {} cm: {}".format(MIN_COVER_CM, len(out)))

out["year"] = pd.to_datetime(out["date"], errors="coerce").dt.year
out = out[out["soc"] > 0]
if MAX_SOC is not None:
    n0 = len(out)
    out = out[out["soc"] <= MAX_SOC]
    print("dropped {} profiles above {} g/kg".format(n0 - len(out), MAX_SOC))
if MIN_YEAR is not None:
    out = out[out["year"] >= MIN_YEAR]
    print("after year >= {}: {}".format(MIN_YEAR, len(out)))

# identical coordinates (several profiles at one spot) -> keep one, so train/test don't leak
n0 = len(out)
out = out.drop_duplicates(subset=["lon", "lat"])
print("dropped {} duplicate locations".format(n0 - len(out)))

out = out[["profile_id", "lon", "lat", "soc", "year", "country", "licence"]].reset_index(drop=True)
out.to_csv(OUT, index=False)

print("\nSAVED", OUT, "| samples:", len(out))
print("SOC summary (should be g/kg; mineral topsoil typically ~5-60):")
print(out["soc"].describe().round(2))
print("\nSamples per country (top 10):")
print(out["country"].value_counts().head(10))
print("\nSamples per decade:")
print((out["year"] // 10 * 10).value_counts().sort_index())
print("\nLicences:")
print(out["licence"].value_counts().head(5))