# UK House Price Prediction

I wanted to predict UK house prices, but do it properly -- actually check whether nearby sales are independent (they're not) and whether a "local price" feature is leaking future information into the past (an easy trap, and one most write-ups on this kind of problem just don't bother with).

## Aims

There were two things I wanted out of this:

1. Predict residential sale price for a bounded problem -- Greater London, 2023 to 2025 -- built as a lightweight AVM (automated valuation model).
2. Actually test the two shortcuts everyone takes in this kind of problem, rather than just assuming they're fine. Nearby sales aren't independent observations, so random k-fold CV quietly overstates how good a model is by leaking neighbours between train and test. And a "local price" feature is trivially easy to leak across time if you compute it over the whole dataset before splitting it. I wanted both of these measured, not just avoided by convention -- see the Moran's I results below, and the leakage rule in `PLAN.md`.

## What's been done

- Data comes from Land Registry Price Paid records joined to ONS postcode coordinates (`data_prep.py`), filtered down to residential arm's-length sales and outlier-trimmed.
- The EDA (`01_eda.ipynb`) covers missingness, price distribution, and a genuinely interesting find: a real, dateable anomaly from the March/April 2025 stamp-duty rush. Also property attribute breakdowns and the reasoning behind the H3 hex resolution I settled on.
- Core spatial features (`features.py`): H3 hex indexing at two resolutions, distance to centre, hex-level trailing price and count, a neighbour-hex spatial lag, and a shrinkage-smoothed hex target encoding. All of it computed strictly from sales *before* the one being predicted, never over the whole dataset and split afterwards.
- I ran a whole documented improvement audit (`PLAN.md`) -- eight further feature and model ideas, each measured against the same evaluation, staying within the same data boundary (no new external datasets). Some of it stuck: multi-resolution hex features, hex target encoding, street/named-property features (though those turned out research-only, more on that below), a tuned XGBoost, LightGBM, prediction intervals. Some of it didn't: a coarser hex resolution, a wider neighbour ring, and, the one that took the most actual work, geographically weighted regression. I wrote that result up anyway instead of quietly dropping it, since it didn't help.
- The modelling notebook (`02_modelling.ipynb`) compares four models -- linear, GAM, XGBoost, LightGBM -- under 5-fold spatial block CV, checks Moran's I on residuals before and after the hex features (and again for the actual best model), runs SHAP, and has predicted-vs-actual scatter plots plus a spatial map of residuals, so the autocorrelation story is something you can see rather than just a number you have to trust.
- There's a working tool too (`predict.py`, walked through in `03_predict.ipynb`): give it a postcode, property type, and tenure, and it gives back a predicted price and an 80% interval, via a CLI. I wanted a range back, not a bare number, and I checked whether that interval's coverage actually holds up rather than assuming it does.

## Hex-price assumption

The hex trailing price, count, and neighbour-lag features are standing in for whatever actually drives value in an area -- schools, transit, amenities, general desirability. The aggregate has already absorbed all of that, so there's no need to name every factor individually. That's the whole point of the H3/spatial-lag approach, and it's why I didn't bring in any external datasets for v1 (see `PLAN.md`). It breaks down wherever what's different about one specific property or street matters more than the area around it, which is exactly why the street-level features in the improvement audit helped as much as they did, and it's what the limitations below are really about.

## Performance

I track two feature sets separately (`src/models.py`): a research set with everything that measurably helps, and a serving set that drops street/address-level features, since `predict.py` only ever gets a postcode, not a specific house.

5-fold spatial block CV, log10(price), research feature set:

| model    | RMSE  | MAE   | R²    |
|----------|-------|-------|-------|
| linear   | 0.155 | 0.113 | 0.630 |
| GAM      | 0.153 | 0.111 | 0.638 |
| XGBoost  | 0.147 | 0.105 | 0.668 |
| LightGBM | 0.147 | 0.106 | 0.667 |

The serving feature set (XGBoost, no address-level features) comes in at RMSE 0.153, MAE 0.110, R² 0.638. That's what `predict.py` actually runs.

Moran's I on out-of-fold residuals (hex-level, k=1 ring weights):

| features                                                         | Moran's I | p (sim) |
|-------------------------------------------------------------------|-----------|---------|
| baseline (no hex features, linear)                                 | 0.723     | 0.001   |
| + hex features (linear)                                            | 0.478     | 0.001   |
| full serving model residuals (XGBoost, postcode-only features)     | 0.261     | 0.001   |
| full research model residuals (XGBoost, incl. street features)     | 0.244     | 0.001   |

Hex features roughly halve residual spatial autocorrelation on their own, and the full feature set with XGBoost cuts it further, down to about a third of where it started. It's still significant at p=0.001 though, so there's real spatial structure left over. I did try to close that gap with GWR, and it didn't clear the bar I'd set for keeping it (see `PLAN.md`).

`predict.py` also returns an 80% prediction interval, from two extra XGBoost quantile models. I measured the actual coverage on the same spatial CV and got about 78%, close enough to the 80% target that a heavier conformal-calibration setup didn't feel necessary.

SHAP on the XGBoost model puts tenure (leasehold vs freehold), distance to centre, and property type at the top, with the street-level and hex target-encoded features close behind.

### Honest scoring

If I had to put a number on this: 6 out of 10 as an actual price-prediction tool, 8 or 9 out of 10 as an example of doing this kind of problem properly. Those feel like genuinely different questions to me.

The process earns the higher mark. Leakage-safe features that are actually tested rather than just asserted, real spatial CV instead of the random k-fold that would have made everything look better than it is, diagnostics I measured instead of assumed, and a negative result (GWR) I wrote up instead of hiding.

The 6 is about the actual accuracy. An R² around 0.64-0.67 and a typical error of 25-30% of price is a real gap from something I'd hand a homeowner as a number to trust. And the biggest reason for that gap isn't something more feature engineering on this dataset fixes: there's no property condition, floor area, or renovation state anywhere in Land Registry data, and that's probably the single biggest thing missing for predicting an individual property, as opposed to "a type of property in an area," which this does reasonably well.

## Limitations

- Residual spatial autocorrelation is reduced, not gone. There's location signal here (condition, exact street, school catchment) this data just doesn't have, and GWR didn't close what's left.
- No property condition, renovation, or floor-area data anywhere -- Price Paid Data doesn't carry it. Probably the single biggest ceiling on accuracy in this project.
- Street-level and named-property signal clearly helps (research model, R² 0.607 to 0.656 on its own) but it's not in the served model, since a postcode alone doesn't tell you the street or house.
- One region, one time window: Greater London, 2023 to 2025. No idea whether any of this holds up elsewhere without refitting the hex features on local data.
- The hex-price proxy assumes the local aggregate is a decent stand-in for whatever's actually driving value -- schools, transit, amenities, general desirability. It falls down wherever what's different within a hex matters more than the hex itself.
- The prediction intervals are wide, often close to double the lower bound. That's an honest reflection of R² still sitting around 0.64, not a bug, but it's not a substitute for a real valuation either.
- "Now" for the served model means the end of the training window, 31 December 2025, not real time. Prices aren't being refreshed live.

## If this were picked back up to actually use

Roughly the order I'd actually tackle these in, based on what the audit showed rather than a generic wishlist:

1. Property condition and size data. Joining the UK's EPC register would give floor area and construction age, but it needs address-level matching against Land Registry records, a known and genuinely fiddly UK open-data linkage problem, not a simple join. Still the highest-leverage thing on this list -- nothing else closes the "individual property" gap the way real size and condition data would.
2. Actually serve the street-level features. The single biggest measured win in the whole audit (R² +0.049) is sitting unused in production because `predict.py` only takes a postcode. Extending it to accept a specific address, falling back to postcode-only when one isn't given, would let the deployed tool use what's already been built.
3. POI and amenity joins -- schools, transit (PTAL), crime, green space. I deliberately scoped these out of v1 to keep things bounded. I'd test them individually rather than as one big effort; distance to the nearest station is the cheapest to try, and the hex-price proxy is probably already absorbing a good chunk of the rest anyway.
4. A retraining pipeline. Right now `--train` is a manual step and "now" is frozen at the last training date. Real use needs this on a schedule, plus some way to notice when the model's assumptions (like the SDLT-rush pattern) have gone stale.
5. More than one region or time window, to actually find out whether any of this generalises, or whether the hex features need refitting from scratch every time.
6. Revisit conformal calibration (`mapie`) for the prediction interval, but only if measured coverage ever drifts meaningfully below what I'm seeing now (~78%). Native XGBoost quantiles already clear the bar I set in `PLAN.md`, so this isn't urgent.

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
├── tests/
└── PLAN.md
```
