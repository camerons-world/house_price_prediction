# UK House Price Prediction

Predicting UK house prices, but actually dealing with the fact that nearby sales aren't independent observations -- most write-ups on this kind of problem ignore spatial autocorrelation and time-based leakage entirely.

Region and time window: Greater London, 2023-01 to 2025-12. Data is Land Registry price paid records joined to ONS postcode coordinates.

## Approach

1. Filter to residential, arm's-length sales (category A, excludes commercial/repossessions), trim price outliers (floor + top 0.1%).
2. Index each sale to an H3 hex (res 8, ~85 median sales/hex over the window) and compute distance-to-centre (haversine from Charing Cross).
3. Hex-level trailing median price, trailing sale count, and a neighbour-hex spatial lag (k=1 ring average) -- all computed strictly from sales *before* the one in question, never over the whole dataset then split. See `features.py`.
4. Three models on log10(price): linear regression, a GAM (factor terms on property attributes, smooth terms on distance/hex features), and XGBoost.
5. All metrics come from 5-fold spatial block CV, grouped by a coarser hex (res 6) so a held-out fold is a contiguous region rather than scattered points with leaky neighbours still in training.
6. Moran's I on out-of-fold residuals, before and after adding the hex features, to check whether they're actually absorbing spatial structure.

## Hex-price assumption

The hex trailing price/count/neighbour-lag features stand in for whatever local factors actually drive value in an area (schools, transit, amenities, general desirability) -- the aggregate has already absorbed the effect, so there's no need to name every factor individually. This is the whole point of the H3/spatial-lag approach and it's why there's no external join in v1 (see `PLAN.md`). It fails wherever within-hex variation matters more than location -- condition, exact street, view, renovation state -- none of which is in this data.

## Results

5-fold spatial block CV, log10(price):

| model    | RMSE  | MAE   | R²    |
|----------|-------|-------|-------|
| linear   | 0.167 | 0.122 | 0.570 |
| GAM      | 0.163 | 0.119 | 0.589 |
| XGBoost  | 0.160 | 0.116 | 0.606 |

Moran's I on out-of-fold residuals (hex-level, k=1 ring weights):

| features                    | Moran's I | p (sim) |
|------------------------------|-----------|---------|
| baseline (no hex features)   | 0.727     | 0.001   |
| + hex features                | 0.485     | 0.001   |

Hex features roughly halve residual spatial autocorrelation -- they're doing real work, not just adding noise the tree model happens to fit. But it's still significant at p=0.001 after adding them: there's real spatial structure left on the table (see Limitations).

SHAP on the XGBoost model ranks tenure (leasehold vs freehold), distance-to-centre, and property type as the strongest individual drivers, with the hex trailing price and neighbour lag close behind -- consistent with the Moran's I result that the hex features carry real signal.

Full walkthrough with charts: `notebooks/02_modeling.ipynb`.

## Limitations

- Residual spatial autocorrelation is reduced but not eliminated -- there's location signal (condition, exact street, school catchment) this data can't capture.
- No property condition, renovation, or floor-area data -- Price Paid Data doesn't carry it.
- One region, one time window (Greater London, 2023-2025) -- results won't generalise to other UK regions without re-fitting the hex features on local data.
- The hex-price proxy assumes local aggregate price is a reasonable stand-in for unobserved local factors; see Hex-price assumption above for where that breaks down.

## Future work

Not attempted in v1, see `PLAN.md` for the full reasoning:

- POI/amenity joins (schools, transit, crime, green space) -- would they actually beat the hex-price proxy?
- Full spatial regression (GWR) instead of the lag-feature workaround.
- More than one region/time window.

## Setup

```bash
python3 -m venv house_price_prediction_venv
source house_price_prediction_venv/bin/activate
pip install -r requirements.txt
```

## Structure

```
├── data/            raw/ and processed/ (gitignored)
├── notebooks/       EDA and modeling
├── src/
│   ├── data_prep.py             Land Registry + ONSPD join
│   ├── features.py              hex indexing, time-based aggregates
│   ├── models.py                 linear, GAM, tree ensemble
│   └── spatial_diagnostics.py    Moran's I
├── tests/
└── PLAN.md
```
