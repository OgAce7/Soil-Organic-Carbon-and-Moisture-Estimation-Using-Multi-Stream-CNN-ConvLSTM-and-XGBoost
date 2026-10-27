"""
get_wosis.py -- download WoSIS "latest" organic carbon points from ISRIC's WFS service.

The address and layer name come from the metadata file you downloaded
(OGC:WFS, layer 'wosis_latest_orgc'). I could not test this from my side,
so run step 1 first and read what it prints.

Usage (from the project root, venv active):
    pip install requests geopandas
    python scripts/get_wosis.py caps      # 1) list layers the server offers
    python scripts/get_wosis.py fetch     # 2) download Europe, save data/raw/wosis_orgc_europe.csv
"""
import io
import os
import sys

import requests

BASE = "https://maps.isric.org/mapserv?map=/map/wosis_latest.map"
LAYER = "wosis_latest_orgc"
# Europe bounding box: lon_min, lat_min, lon_max, lat_max
BBOX = (-10.0, 36.0, 30.0, 60.0)
OUT = "data/raw/wosis_orgc_europe.csv"


def caps():
    r = requests.get(BASE, params={"SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetCapabilities"}, timeout=120)
    print("HTTP", r.status_code)
    import re
    names = re.findall(r"<(?:\w+:)?Name>([^<]+)</(?:\w+:)?Name>", r.text)
    print("Layers / names on server:")
    for n in names:
        print("  ", n)
    print("\nLook for the one containing 'orgc' and use it as LAYER (maybe with an 'ms:' prefix).")


def fetch():
    import geopandas as gpd

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tried = []
    for typename in (LAYER, "ms:" + LAYER):
        for fmt in ("application/json; subtype=geojson", "application/json", "GML3"):
            params = {
                "SERVICE": "WFS", "VERSION": "1.1.0", "REQUEST": "GetFeature",
                "TYPENAME": typename, "OUTPUTFORMAT": fmt,
                "BBOX": "{},{},{},{},EPSG:4326".format(BBOX[1], BBOX[0], BBOX[3], BBOX[2]),
            }
            try:
                r = requests.get(BASE, params=params, timeout=600)
                if r.status_code != 200 or b"ExceptionReport" in r.content[:500]:
                    tried.append((typename, fmt, r.status_code, r.text[:150].replace("\n", " ")))
                    continue
                gdf = gpd.read_file(io.BytesIO(r.content))
                if len(gdf) == 0:
                    tried.append((typename, fmt, "empty", ""))
                    continue
                gdf["lon"] = gdf.geometry.x
                gdf["lat"] = gdf.geometry.y
                gdf.drop(columns="geometry").to_csv(OUT, index=False)
                print("OK:", typename, fmt, "->", OUT)
                print("rows:", len(gdf))
                print("columns:", list(gdf.columns))
                print(gdf.drop(columns="geometry").head())
                return
            except Exception as e:  # keep trying other combos
                tried.append((typename, fmt, "error", str(e)[:150]))
    print("All attempts failed:")
    for t in tried:
        print("  ", t)
    print("\nRun `python scripts/get_wosis.py caps` and paste me the output.")


if __name__ == "__main__":
    {"caps": caps, "fetch": fetch}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()