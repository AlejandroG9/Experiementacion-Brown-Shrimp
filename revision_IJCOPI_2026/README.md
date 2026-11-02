# Corrected experiment – Random Forest vs Boosted Trees (brown shrimp catch)

Code and per-run results for the revised version of *"Quantitative Analysis of Brown Shrimp Catch Using Random Forest and Boosted Trees"* (IJCOPI).

## Protocol
- 60 independent replicates (seed `1000 + r`) × training proportions 20%–80% (step 10%).
- Within each replicate the **same train/test partition** is used for Random Forest, XGBoost and for raw / Min-Max data (true pairing).
- Min-Max scaler fitted on the training partition only; target kept in original units (kg per haul).
- Equal tuning budget: `RandomizedSearchCV` (15 of 36 configurations, 5-fold CV, MSE) on the training partition only; best model refitted and evaluated once on the test set.
- Mean baseline (`DummyRegressor`).

## Files
| File | Content |
|---|---|
| `experimento.py` | Runs the experiment and writes `resultados.csv` (resumable). |
| `resultados.csv` | Per-run MAE, RMSE, R², CV RMSE, selected hyperparameters and time. |
| `analisis.py` | Wilcoxon signed-rank tests (exact p, rank-biserial r, bootstrap 95% CI, Holm), Friedman tests, hyperparameter frequencies and figures → `salidas/`. |
| `figura_flujo.py` | Flowchart of the protocol (`salidas/fig_flujo.png`). |

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python experimento.py --reps 60 --out resultados.csv   # requires the dataset
python analisis.py resultados.csv                      # works with the published results
```

## Data availability
The **Statistics V5** dataset is **not included**. Catch records come from the Onboard Observer Programme of the Instituto Nacional de Pesca y Acuacultura (INAPESCA, Mexico), provided under official letter RJL/INAPESCA/DGAIPA/012b/2020, which authorises their use exclusively for this research and does not allow public release. Requests for access should be addressed to INAPESCA. Satellite variables come from NASA MODIS-Aqua Level-3 monthly 4 km products (SST: doi:10.5067/MODSA-MO4D9; chlorophyll-a: doi:10.5067/AQUA/MODIS/L3M/CHL/2022).
