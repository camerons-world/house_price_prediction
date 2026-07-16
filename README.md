# UK House Price Prediction

Predicting UK house prices, but actually dealing with the fact that nearby sales aren't independent observations -- most write-ups on this kind of problem ignore spatial autocorrelation and time-based leakage entirely.

Region and time window: Greater London, 2023-01 to 2025-12. Data is Land Registry price paid records joined to ONS postcode coordinates.

Still working through it.

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
