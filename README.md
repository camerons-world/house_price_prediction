# UK House Price Prediction

Predicting UK house prices, but actually dealing with the fact that nearby sales aren't independent observations -- most write-ups on this kind of problem ignore spatial autocorrelation and time-based leakage entirely.

Region and time window: Greater London, 2023-01 to 2025-12. Data is Land Registry price paid records joined to ONS postcode coordinates.

## Approach

1. Filter to residential, arm's-length sales (category A, excludes commercial/repossessions), trim price outliers (floor + top 0.1%).
2. Index each sale to an H3 hex (res 8, ~85 median sales/hex over the window, plus a finer res-9 hex for a tighter local trend) and compute distance-to-centre (haversine from Charing Cross).
3. Hex-level trailing median price, trailing sale count, a neighbour-hex spatial lag (k=1 ring average), and a separately-shrunk expanding-window hex price level (target encoding) -- all computed strictly from sales *before* the one in question, never over the whole dataset then split. See `features.py`.
4. Four models on log10(price): linear regression, a GAM (factor terms on property attributes, smooth terms on distance/hex features), XGBoost (hyperparameters tuned via spatial-CV search), and LightGBM.
5. All metrics come from 5-fold spatial block CV, grouped by a coarser hex (res 6) so a held-out fold is a contiguous region rather than scattered points with leaky neighbours still in training.
6. Moran's I on out-of-fold residuals, before and after adding the hex features, to check whether they're actually absorbing spatial structure.
7. A documented improvement audit (`PLAN.md`) tried eight further feature/model changes within this same data boundary -- some kept (multi-resolution hex, target encoding, tuning), some dropped after measuring no real lift (k=2 neighbour ring, GWR), one (address-level street/named-property features) kept for research only since it needs more than a postcode to serve live.

## Hex-price assumption

The hex trailing price/count/neighbour-lag features stand in for whatever local factors actually drive value in an area (schools, transit, amenities, general desirability) -- the aggregate has already absorbed the effect, so there's no need to name every factor individually. This is the whole point of the H3/spatial-lag approach and it's why there's no external join in v1 (see `PLAN.md`). It fails wherever within-hex variation matters more than location -- condition, exact street, view, renovation state -- none of which is in this data.

## Results

Two feature sets are tracked separately (`src/models.py`): **research** (everything that measurably helps, used for model comparison/SHAP) and **serving** (research minus address-level features, since `src/predict.py` only ever gets a postcode, not a specific street or house).

5-fold spatial block CV, log10(price), research feature set:

| model    | RMSE  | MAE   | R²    |
|----------|-------|-------|-------|
| linear   | 0.155 | 0.113 | 0.630 |
| GAM      | 0.153 | 0.111 | 0.638 |
| XGBoost  | 0.147 | 0.105 | 0.668 |
| LightGBM | 0.147 | 0.106 | 0.667 |

Serving feature set (XGBoost, no address-level features): RMSE 0.153, MAE 0.110, R² 0.638 -- this is what `predict.py` actually runs.

Moran's I on out-of-fold residuals (hex-level, k=1 ring weights):

| features                                      | Moran's I | p (sim) |
|------------------------------------------------|-----------|---------|
| baseline (no hex features, linear)              | 0.723     | 0.001   |
| + hex features (linear)                         | 0.478     | 0.001   |
| full serving model residuals (XGBoost, postcode-only features)  | 0.261     | 0.001   |
| full research model residuals (XGBoost, incl. street features)  | 0.244     | 0.001   |

Hex features roughly halve residual spatial autocorrelation on their own; the full engineered feature set with XGBoost cuts it further, to about a third of the original baseline. Still significant at p=0.001 -- real spatial structure remains (see Limitations), and a dedicated attempt to close that gap with geographically weighted regression was tried and didn't clear the bar (see `PLAN.md`'s Future Work).

`predict.py` also returns an 80% prediction interval (two extra XGBoost quantile models, native `reg:quantileerror`), not just a point estimate -- measured empirical coverage on the same spatial CV is ~78%, close to the 80% nominal target.

SHAP on the XGBoost model (full research feature set) ranks tenure (leasehold vs freehold), distance-to-centre, and property type as the strongest individual drivers, with street-level and hex target-encoded features close behind -- consistent with the Moran's I result that they carry real signal.

Full walkthrough with charts: `notebooks/02_modeling.ipynb`; interactive postcode tool: `notebooks/03_predict.ipynb`.

## Limitations

- Residual spatial autocorrelation is reduced but not eliminated -- there's location signal (condition, exact street, school catchment) this data can't capture, and GWR (tried) didn't close the remaining gap.
- No property condition, renovation, or floor-area data -- Price Paid Data doesn't carry it.
- Street-level and named-property signal measurably improves accuracy (research model) but isn't in the served model -- a postcode alone doesn't specify a street or house, so `predict.py` can't use it.
- One region, one time window (Greater London, 2023-2025) -- results won't generalise to other UK regions without re-fitting the hex features on local data.
- The hex-price proxy assumes local aggregate price is a reasonable stand-in for unobserved local factors; see Hex-price assumption above for where that breaks down.
- Prediction intervals are wide (often close to double the lower bound) -- an honest reflection of R²~0.64, not a bug, but not a substitute for a real valuation.

## Future work

Not attempted in v1, see `PLAN.md` for the full reasoning (including a documented negative result: GWR was tried and didn't beat the existing hex-lag approach):

- POI/amenity joins (schools, transit, crime, green space) -- would they actually beat the hex-price proxy?
- Serving street-level features would need `predict.py` to accept a specific address, not just a postcode.
- More than one region/time window.

## Setup

```bash
python3 -m venv house_price_prediction_venv
source house_price_prediction_venv/bin/activate
pip install -r requirements.txt
```

## Structure

```
├── data/            raw/ and processed/ (gitignored -- incl. trained model/quantile
│                    models and hex snapshots saved by `python -m src.predict --train`)
├── notebooks/       EDA, modeling, and the postcode predictor walkthrough
├── src/
│   ├── data_prep.py             Land Registry + ONSPD join
│   ├── features.py              hex indexing, time-based aggregates, address features
│   ├── models.py                 linear, GAM, xgboost, lightgbm, tuning, quantile models
│   ├── spatial_diagnostics.py    Moran's I, spatial block CV
│   └── predict.py                postcode -> price CLI/tool (`python -m src.predict`)
├── tests/
└── PLAN.md
```
