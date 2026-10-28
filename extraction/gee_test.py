import ee

ee.Initialize(project='soil-organic-carbon-estimation')

datasets = {
    'MOD13Q1 (EVI)': 'MODIS/061/MOD13Q1',
    'MCD12Q2 (phenology)': 'MODIS/061/MCD12Q2',
    'SRTM (DEM)': 'USGS/SRTMGL1_003',
}

for name, asset in datasets.items():
    if 'SRTM' in asset:
        img = ee.Image(asset)
        print(name, '-> OK, bands:', img.bandNames().getInfo())
        continue
    ic = ee.ImageCollection(asset)
    first = ee.Date(ic.aggregate_min('system:time_start')).format('YYYY-MM-dd').getInfo()
    last = ee.Date(ic.aggregate_max('system:time_start')).format('YYYY-MM-dd').getInfo()
    print(name, '->', first, 'to', last, '| bands:', ic.first().bandNames().getInfo())