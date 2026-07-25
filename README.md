# UK House Price Prediction

Predicting UK house prices -- actually checking whether nearby sales are independent (they're not) and whether a "local price" feature leaks future information into the past (an easy trap most write-ups on this kind of problem skip).

## Aims

1. Predict residential sale price for a bounded problem -- Greater London, 2023 to 2025 -- as a lightweight AVM (automated valuation model).
2. Measure the two shortcuts everyone takes here instead of assuming they're fine: random k-fold CV leaks neighbouring sales between train and test, and a "local price" feature is trivially easy to leak across time if computed over the whole dataset before splitting. See the Moran's I results below and the leakage rule enforced in `features.py`.

## What's been done

- Data: Land Registry Price Paid records joined to ONS postcode coordinates (`data_prep.py`), filtered to residential arm's-length sales, outlier-trimmed.
- EDA (`01_eda.ipynb`): missingness, price distribution, a dateable anomaly from the March/April 2025 stamp-duty rush, property attribute breakdowns, and the reasoning behind the H3 hex resolution.
- Spatial features (`features.py`): H3 hex indexing at two resolutions, distance to centre, hex-level trailing price/count, a neighbour-hex spatial lag, and a shrinkage-smoothed hex target encoding -- all computed strictly from sales *before* the one being predicted.
- A documented improvement audit -- eight further feature/model ideas measured against the same evaluation. Kept: multi-resolution hex features, hex target encoding, street/named-property features (research-only, more below), tuned XGBoost, LightGBM, prediction intervals. Dropped: a coarser hex resolution, a wider neighbour ring, and geographically weighted regression -- written up anyway since it didn't help.
- Modelling (`02_modelling.ipynb`): linear, GAM, XGBoost, LightGBM under 5-fold spatial block CV, Moran's I on residuals before/after hex features, SHAP, predicted-vs-actual and residual maps.
- A working tool (`predict.py`, walked through in `03_predict.ipynb`): postcode, property type, and tenure in, a predicted price and 80% interval out, via CLI. Interval coverage is checked, not assumed.

## Hex-price assumption

The hex trailing price, count, and neighbour-lag features stand in for whatever actually drives value in an area -- schools, transit, amenities, general desirability -- without naming each factor individually. That's the point of the H3/spatial-lag approach, and why no external datasets were needed for v1. It breaks down wherever one specific property or street matters more than the area around it, which is exactly why the street-level features helped as much as they did (see limitations).

## Performance

Two feature sets, tracked separately (`src/models.py`): a research set with everything that measurably helps, and a serving set that drops street/address-level features, since `predict.py` only ever gets a postcode.

5-fold spatial block CV, log10(price), research feature set:

| model    | RMSE  | MAE   | R²    |
|----------|-------|-------|-------|
| linear   | 0.155 | 0.113 | 0.630 |
| GAM      | 0.153 | 0.111 | 0.638 |
| XGBoost  | 0.147 | 0.105 | 0.668 |
| LightGBM | 0.147 | 0.106 | 0.667 |

Serving feature set (XGBoost, no address-level features): RMSE 0.153, MAE 0.110, R² 0.638 -- what `predict.py` actually runs.

Moran's I on out-of-fold residuals (hex-level, k=1 ring weights):

| features                                                         | Moran's I | p (sim) |
|-------------------------------------------------------------------|-----------|---------|
| baseline (no hex features, linear)                                 | 0.723     | 0.001   |
| + hex features (linear)                                            | 0.478     | 0.001   |
| full serving model residuals (XGBoost, postcode-only features)     | 0.261     | 0.001   |
| full research model residuals (XGBoost, incl. street features)     | 0.244     | 0.001   |

Hex features roughly halve residual spatial autocorrelation on their own; the full feature set with XGBoost cuts it to about a third of where it started. Still significant at p=0.001, so real spatial structure remains -- GWR didn't close that gap enough to justify keeping it.

`predict.py`'s 80% prediction interval (two XGBoost quantile models) measures about 78% actual coverage on the same spatial CV -- close enough that a heavier conformal-calibration setup didn't feel necessary.

SHAP on the XGBoost model puts tenure (leasehold vs freehold), distance to centre, and property type at the top, with street-level and hex target-encoded features close behind.

## Limitations

- Residual spatial autocorrelation is reduced, not gone. Condition, exact street, and school catchment aren't in this data, and GWR didn't close what's left.
- No property condition, renovation, or floor-area data -- Price Paid Data doesn't carry it. Probably the single biggest ceiling on accuracy here.
- Street-level/named-property signal clearly helps (research model, R² 0.607 to 0.656 on its own) but isn't served, since a postcode alone doesn't give a street or house.
- One region, one time window: Greater London, 2023 to 2025. No idea whether this holds up elsewhere without refitting the hex features.
- The hex-price proxy assumes the local aggregate is a decent stand-in for what actually drives value. It falls down wherever one property differs from its hex more than the hex itself matters.
- Prediction intervals are wide, often close to double the lower bound -- an honest reflection of R² around 0.64, not a bug, but not a substitute for a real valuation.
- "Now" for the served model means the end of the training window, 31 December 2025, not real time.

## If this were picked back up to actually use

Roughly the order I'd tackle these in, based on what the audit showed:

1. Property condition and size data. The UK's EPC register would give floor area and construction age, but needs fiddly address-level matching against Land Registry records. Still the highest-leverage gap on this list.
2. Actually serve the street-level features. The audit's single biggest win (R² +0.049) sits unused because `predict.py` only takes a postcode -- extending it to accept an address, falling back to postcode-only, would let the deployed tool use what's already built.
3. POI and amenity joins -- schools, transit (PTAL), crime, green space. Scoped out of v1 to keep things bounded; I'd test them individually, starting with distance to the nearest station.
4. A retraining pipeline. `--train` is a manual step and "now" is frozen at the last training date -- real use needs this on a schedule, plus a way to notice when assumptions (like the SDLT-rush pattern) go stale.
5. More than one region or time window, to find out whether any of this generalises.
6. Revisit conformal calibration (`mapie`) for the prediction interval, but only if measured coverage drifts meaningfully below what I'm seeing now (~78%). Native XGBoost quantiles already clear the bar, so this isn't urgent.

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
├── notebooks/       EDA, modelling, and the postcode predictor walkthrough
├── src/
│   ├── data_prep.py             Land Registry + ONSPD join
│   ├── features.py              hex indexing, time-based aggregates, address features
│   ├── models.py                 linear, GAM, xgboost, lightgbm, tuning, quantile models
│   ├── spatial_diagnostics.py    Moran's I, spatial block CV
│   └── predict.py                postcode -> price CLI/tool (`python -m src.predict`)
└── tests/
```
