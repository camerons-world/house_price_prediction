# Spatially-Aware UK House Price Prediction

Status: **planned, not yet implemented** — see [PLAN.md](PLAN.md) for the full project plan, scope boundaries, and design decisions.

## Overview

A lightweight Automated Valuation Model (AVM) predicting UK residential sale prices from property attributes and location, with explicit treatment of two issues most portfolio regression projects skip:

- **Spatial autocorrelation** in residuals (diagnosed via Moran's I, addressed via H3 hex features)
- **Time-based leakage** in location-derived features (hex-level historical price is built with a strict no-lookahead, trailing-window rule)

This is a statistical-rigor showcase, not a deployment showcase — no Docker/API/serving layer here (see companion project `project-2-flood-risk-classifier` for that).

## Data

- HM Land Registry Price Paid Data (UK), joined to the ONS Postcode Directory (ONSPD) for postcode → lat/long.
- Scoped to a single region and a 2-3 year transaction window (TBD — see PLAN.md).

## Repo Structure

```
├── data/            raw/ and processed/ (gitignored — not committed)
├── notebooks/       EDA and modeling walkthroughs
├── src/
│   ├── data_prep.py           Land Registry + ONSPD join
│   ├── features.py            H3 hex indexing, time-based aggregates
│   ├── models.py               linear, GAM, tree ensemble
│   └── spatial_diagnostics.py  Moran's I
├── tests/
├── requirements.txt
└── PLAN.md
```

## Setup

```bash
python3 -m venv house_price_prediction_venv
source house_price_prediction_venv/bin/activate
pip install -r requirements.txt
```

## Results

_TBD — to be filled in as modeling progresses._

## Limitations & Future Work

See "Future Work" in [PLAN.md](PLAN.md): POI/amenity joins (schools, transit, crime), full spatial regression (GWR), and multi-region extension are explicitly out of scope for v1.
