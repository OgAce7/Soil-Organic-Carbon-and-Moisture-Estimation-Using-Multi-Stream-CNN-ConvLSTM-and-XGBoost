# CNN-LSTM Soil Organic Carbon (SOC) Prediction, Europe

Predicts topsoil (0-30 cm) soil organic carbon in g/kg at sample points, using a CNN for static terrain/climate windows and an LSTM for vegetation time series.

Data: ISRIC WoSIS (SOC samples), MODIS (EVI, phenology), SRTM (terrain), WorldClim (climate). No data or trained model ships with the code: you build both by following the steps below.

---

## 1. Prerequisites

- **Python 3.10 - 3.12**
- A **Google Earth Engine** account registered for a Google Cloud project (you need the project ID)
- Internet access (WoSIS download and Earth Engine)
- No GPU needed. Everything runs on CPU.

## 2. Create the virtual environment

Open a terminal in the project folder (the one containing `config.py`).

**Windows (PowerShell or cmd):**
```
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the start of the prompt. Activate it every time you open a new terminal.

## 3. Install the packages

```
python -m pip install --upgrade pip
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn tqdm requests geopandas earthengine-api
pip freeze > requirements.txt
```

On a fresh machine you can instead run `pip install -r requirements.txt`.

## 4. Create the folders

```
mkdir scripts extraction data data\raw data\train_test_idx model log results
```
(macOS/Linux: `mkdir -p scripts extraction data/raw data/train_test_idx model log results`)

`extraction/build_pickles.py` also creates the `data`, `model` and `log` folders if they are missing.

## 5. Put the files in place

| File | Goes in |
|---|---|
| `config.py`, `utils.py`, `models.py`, `data_helper.py`, `train.py`, `pred.py` | project root (original code) |
| `get_wosis.py`, `clean_soc.py`, `plot_soc.py`, `plot_scatter.py` | `scripts/` |
| `gee_test.py`, `build_pickles.py` | `extraction/` |

**Always run every command from the project root** (the folder containing `config.py`), because the code uses plain `import config`.

## 6. Authenticate Earth Engine (once)

```
earthengine authenticate
```
Follow the browser prompt. Then edit `extraction/gee_test.py` and replace `YOUR_PROJECT_ID` with your real Google Cloud project ID, and check it works:

```
python extraction/gee_test.py
```
You should see date ranges for MOD13Q1 (EVI) and MCD12Q2 (phenology) and the SRTM band list. If you get "Project ... not found", the project ID is wrong or the Earth Engine API is not enabled for that project.

## 7. Build the data (in this order)

**Step 1 - download WoSIS organic carbon points for Europe**
```
python scripts/get_wosis.py fetch
```
Saves `data/raw/wosis_orgc_europe.csv` (about 30,000 layer rows). If it fails, run `python scripts/get_wosis.py caps` to list the server's layers.

**Step 2 - clean to one topsoil value per profile**
```
python scripts/clean_soc.py
```
Saves `data/raw/samples_clean.csv` (about 7,000 profiles). Check the printed SOC summary: values should be in g/kg, with a median around 15-20.

**Step 3 (optional) - overview figure of the samples**
```
python scripts/plot_soc.py
```
Saves `data/raw/soc_overview.png`.

**Step 4 - extract model inputs from Earth Engine**
```
python extraction/build_pickles.py --project YOUR_PROJECT_ID --n 300
```
Run the 300-point test first. Then a bigger run:
```
python extraction/build_pickles.py --project YOUR_PROJECT_ID --n 2000
```
(`--n 0` uses all points; slower.) This writes `data/samples_data.csv`, the three `.pkl` input files and the train/test index files. About 2,000 points took roughly 4 minutes. Chunks are cached in `data/raw/gee_cache/`, so an interrupted run can be resumed by repeating the command.

Expected shapes: CNN input `(N, 10, 7, 7)`, EVI `(N, 60, 1)`, phenology `(N, 60, 11)`.

## 8. Edit three things before training

1. **`config.py`**: change `epochs = 1000` to `epochs = 100`. 1000 is far too many for this data and overfits.
2. **`train.py`**: make it save only the best model. Near the end of the evaluation function, replace
   ```python
   torch.save(model.state_dict(), cfg.model_save_pth)
   ```
   with
   ```python
   if r2 > best_r2:
       best_rmse, best_mae, best_r2, best_epoch = rmse, mae, r2, epoch
       torch.save(model.state_dict(), cfg.model_save_pth)
       print('  -> new best model saved (epoch {})'.format(epoch))
   ```
3. **`pred.py`**: in `main()`, replace
   ```python
   model_name = 'CNN-LSTM_evi_lsp'
   model = get_model(model_name=model_name, model_save_pth='./model/CNN-LSTM_params.pth')
   ```
   with
   ```python
   model_name = cfg.model_name
   model = get_model(model_name=model_name, model_save_pth=cfg.model_save_pth)
   ```

## 9. Train, predict, plot

```
python train.py
```
It prints the data shapes and waits: check them, then press **Enter**. Every 10 epochs it prints train and test RMSE, MAE and R2. The model is saved to `model/CNN-LSTM_evi_lsp_1.pth`.

```
python pred.py
```
Scores the saved model on the held-out test set (should match the best test R2 from training).

```
python scripts/plot_scatter.py
```
Writes `results/scatter_pred_vs_obs.png` and `results/predictions_test.csv`.

## 10. Comparing model variants (optional)

Change `model_name` in `config.py` to one of: `CNN`, `LSTM_evi`, `LSTM_lsp`, `LSTM_evi_lsp`, `CNN-LSTM_evi`, `CNN-LSTM_lsp`, `CNN-LSTM_evi_lsp`. Then re-run `train.py` and `pred.py`. Each variant saves its own weights file, so nothing is overwritten. Note `plot_scatter.py` only supports `CNN-LSTM_evi_lsp`.

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'config'` | Run from the project root, not from inside `scripts/` or `extraction/`. |
| `Project 'projects/YOUR_PROJECT_ID' not found` | Put your real Google Cloud project ID in the command / script. |
| `FileNotFoundError: ./model/CNN-LSTM_params.pth` | Apply the `pred.py` edit in section 8. |
| `torch.save` fails with "No such file or directory" | Create the `model` folder (`mkdir model`). |
| `final samples` far below the points requested | Make sure you use the latest `build_pickles.py` (it handles masked pixels with a placeholder value). |
| Earth Engine requests keep failing | Lower `CHUNK` at the top of `build_pickles.py` (try 50) and re-run; cached chunks are reused. |
| Test R2 much lower than training R2 | Normal here: the model overfits after about 40 epochs. Keep the best-checkpoint edit. |

## 12. Folder layout

```
DL Project/
├── .venv/
├── requirements.txt
├── config.py            settings (model name, paths, hyper-parameters)
├── utils.py             data loading helpers
├── models.py            the three network classes (ConvNet, SimpleLSTM, CovLSTM)
├── data_helper.py
├── train.py             training loop
├── pred.py              evaluation on the test set
├── scripts/             get_wosis.py, clean_soc.py, plot_soc.py, plot_scatter.py
├── extraction/          gee_test.py, build_pickles.py
├── data/                raw/ + samples_data.csv + 3 input .pkl files + train_test_idx/
├── model/               saved weights (.pth)
├── log/
└── results/             scatter plot and test predictions
```

## 13. Data credits and licences

- SOC samples: ISRIC WoSIS latest, Batjes et al. (2024), Earth System Science Data 16, 4735, https://doi.org/10.5194/essd-16-4735-2024. Licences vary per profile (CC BY 3.0/4.0, CC BY-NC 4.0, OGL, public domain). Keep use non-commercial and cite ISRIC.
- MODIS MOD13Q1 v061 (EVI) and MCD12Q2 v061 (phenology), NASA, via Google Earth Engine.
- SRTM 30 m elevation and WorldClim v1 bioclimatic variables, via Google Earth Engine.
