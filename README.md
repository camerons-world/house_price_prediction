# UK House Price Prediction

Predicting UK house prices, with a deliberate investigation into spatial autocorrelation (nearby sales aren't independent observations) and time-based leakage in the underlying data.
## Aims

Two things, one practical and one methodological:

1. **Predict residential sale price** for a bounded problem -- Greater London, 2023-01 to 2025-12 -- framed as a lightweight AVM (automated valuation model).
2. **Actually test, rather than assume, that the two usual shortcuts in this kind of problem are safe.** Nearby sales aren't independent observations (random k-fold CV overstates accuracy by leaking neighbours between train/test), and a "local price" feature is trivially easy to leak across time if it's computed over the whole dataset before splitting. Both are enforced and *measured*, not just avoided by convention -- see the Moran's I results below and the leakage rule in `PLAN.md`.

## What's been done

- **Data**: Land Registry Price Paid Data joined to ONS postcode coordinates (`data_prep.py`), filtered to residential arm's-length sales, outlier-trimmed.
- **EDA** (`01_eda.ipynb`): missingness, price distribution, a real dateable anomaly in the data (the March/April 2025 stamp-duty rush), property attribute breakdowns, and the empirical justification for the H3 hex resolution used throughout.
- **Core spatial features** (`features.py`): H3 hex indexing (res 8, plus a finer res 9), distance-to-centre, hex-level trailing price/count, a neighbour-hex spatial lag, and a shrinkage-smoothed hex target encoding -- all computed strictly from sales *before* the one being predicted, never over the whole dataset then split.
- **A documented improvement audit** (`PLAN.md`): eight further feature/model changes tried and measured against the same evaluation, within the same data boundary (no new external datasets). Kept: multi-resolution hex features, hex target encoding, street/named-property features (research-only, see Limitations), XGBoost tuning, LightGBM, prediction intervals. Dropped after measuring no real lift: a coarser hex resolution, a wider neighbour ring, and -- the most involved experiment -- geographically weighted regression (GWR), documented as a negative result rather than silently discarded.
- **Modelling** (`02_modeling.ipynb`): four models compared (linear, GAM, XGBoost, LightGBM) under 5-fold spatial block CV, Moran's I on residuals before/after hex features *and* for the actual best model, SHAP feature importance, predicted-vs-actual scatter plots, and a spatial choropleth of residuals (baseline vs. full model) to make the autocorrelation story visible, not just a statistic.
- **A working tool** (`predict.py`, walked through in `03_predict.ipynb`): postcode + property type + tenure in, a predicted price *and* an 80% prediction interval out, via a CLI (`python -m src.predict`). Point estimate plus interval, not a bare number -- interval coverage is measured, not assumed.

## Hex-price assumption

The hex trailing price/count/neighbour-lag features stand in for whatever local factors actually drive value in an area (schools, transit, amenities, general desirability) -- the aggregate has already absorbed the effect, so there's no need to name every factor individually. This is the whole point of the H3/spatial-lag approach and it's why there's no external join in v1 (see `PLAN.md`). It fails wherever within-hex variation matters more than location -- condition, exact street, view, renovation state -- which is exactly why the street-level features in the improvement audit measurably helped (see Performance) and exactly what the Limitations below are about.

## Performance

Two feature sets are tracked separately (`src/models.py`): **research** (everything that measurably helps) and **serving** (research minus street/address-level features, since `predict.py` only ever gets a postcode, not a specific house).

5-fold spatial block CV, log10(price), research feature set:

| model    | RMSE  | MAE   | R²    |
|----------|-------|-------|-------|
| linear   | 0.155 | 0.113 | 0.630 |
| GAM      | 0.153 | 0.111 | 0.638 |
| XGBoost  | 0.147 | 0.105 | 0.668 |
| LightGBM | 0.147 | 0.106 | 0.667 |

Serving feature set (XGBoost, no address-level features): RMSE 0.153, MAE 0.110, R² 0.638 -- this is what `predict.py` actually runs.

Moran's I on out-of-fold residuals (hex-level, k=1 ring weights):

| features                                                         | Moran's I | p (sim) |
|-------------------------------------------------------------------|-----------|---------|
| baseline (no hex features, linear)                                 | 0.723     | 0.001   |
| + hex features (linear)                                            | 0.478     | 0.001   |
| full serving model residuals (XGBoost, postcode-only features)     | 0.261     | 0.001   |
| full research model residuals (XGBoost, incl. street features)     | 0.244     | 0.001   |

Hex features roughly halve residual spatial autocorrelation on their own; the full engineered feature set with XGBoost cuts it further, to about a third of the original baseline. Still significant at p=0.001 -- real spatial structure remains, and a dedicated attempt to close that gap with GWR was tried and didn't clear the bar (see `PLAN.md`).

`predict.py` also returns an 80% prediction interval (two extra XGBoost quantile models, native `reg:quantileerror`) -- measured empirical coverage on the same spatial CV is ~78%, close to the 80% nominal target, so no need yet for a heavier conformal-calibration approach.

SHAP on the XGBoost model ranks tenure (leasehold vs freehold), distance-to-centre, and property type as the strongest individual drivers, with street-level and hex target-encoded features close behind.

## Limitations

- Residual spatial autocorrelation is reduced but not eliminated -- there's location signal (condition, exact street, school catchment) this data can't capture, and GWR (tried) didn't close the remaining gap.
- No property condition, renovation, or floor-area data -- Price Paid Data doesn't carry it. Plausibly the single biggest accuracy ceiling in this project.
- Street-level and named-property signal measurably improves accuracy (research model, R² 0.607 -> 0.656 on its own) but isn't in the served model -- a postcode alone doesn't specify a street or house, so `predict.py` can't use it.
- One region, one time window (Greater London, 2023-2025) -- results won't generalise to other UK regions without re-fitting the hex features on local data.
- The hex-price proxy assumes local aggregate price is a reasonable stand-in for unobserved local factors (schools, transit, amenities, general desirability) -- it fails wherever within-hex variation matters more than location.
- Prediction intervals are wide (often close to double the lower bound) -- an honest reflection of R²~0.64, not a bug, but not a substitute for a real valuation.
- "Now" for the served model means the end of the training window (2025-12-31), not real time -- prices aren't refreshed live.

## If this were picked back up to actually use

Roughly in priority order, based on what the audit actually showed rather than a generic wishlist:

1. **Property condition/size data** -- e.g. joining the UK's EPC (Energy Performance Certificate) register for floor area and construction age. Needs address-level matching against Land Registry records (a known, fiddly UK open-data linkage problem, not a simple join), but this is the highest-leverage lever identified: nothing else closes the "individual property" gap the way real size/condition data would.
2. **Serve the street-level features** -- the single biggest measured win (R² +0.049) is sitting unused in production because `predict.py` only takes a postcode. Extending it to accept a specific address (and falling back to the current postcode-only behaviour when one isn't given) would let the deployed tool actually use it.
3. **POI/amenity joins** -- schools, transit (PTAL), crime, green space. `PLAN.md` deliberately scoped these out of v1 to keep it bounded; worth testing individually (transit distance is the cheapest to try, a small static station-location dataset) rather than as one big joint effort, given the hex-price proxy already absorbs a lot of this indirectly.
4. **A retraining/refresh pipeline** -- right now `--train` is a manual step and "now" is frozen at the last training date. Real use needs this on a schedule, plus monitoring for when the model's assumptions (e.g. the SDLT-rush pattern) go stale.
5. **More than one region/time window** -- to know whether any of this generalises, or whether the hex features would need refitting per region from scratch.
6. **Revisit `mapie`/conformal calibration** for the prediction interval if measured coverage ever drifts materially below the ~78% currently seen -- deliberately not done now since native XGBoost quantiles already clear the bar set in `PLAN.md`.

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
