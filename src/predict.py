from __future__ import annotations

import argparse
import json

import h3
import joblib
import pandas as pd

from src import models
from src.data_prep import DATA_PROCESSED, _normalize_postcode, load_onspd_lookup
from src.features import TRAILING_WINDOW, add_distance_to_centre, add_hex_id

HEX_SNAPSHOT_PATH = DATA_PROCESSED / "hex_snapshot.csv"
SNAPSHOT_META_PATH = DATA_PROCESSED / "snapshot_meta.json"


def build_hex_snapshot(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # one fixed reference date instead of models.py's per-row rolling window --
    # this is what "current" local price looks like as of the end of the training data,
    # never real wall-clock today (the model never saw dates past this)
    reference_date = df["date_of_transfer"].max()
    window_start = reference_date - pd.Timedelta(TRAILING_WINDOW)
    recent = df[(df["date_of_transfer"] >= window_start) & (df["date_of_transfer"] < reference_date)]

    snapshot = recent.groupby("hex_id")["price"].agg(hex_trailing_price="median", hex_trailing_count="count")
    citywide_price = recent["price"].median()

    lookup = snapshot["hex_trailing_price"]

    def neighbor_avg(h):
        neighbors = [n for n in h3.grid_disk(h, 1) if n != h]
        vals = lookup.reindex(neighbors).dropna()
        return vals.mean() if len(vals) else lookup[h]

    snapshot["hex_neighbor_lag"] = [neighbor_avg(h) for h in snapshot.index]

    meta = {
        "reference_date": str(reference_date.date()),
        "reference_year": int(reference_date.year),
        "reference_month": int(reference_date.month),
        "citywide_trailing_price": float(citywide_price),
    }
    return snapshot.reset_index(), meta


def train_and_save() -> None:
    df = models.load_dataset()
    pipe = models.fit_full_pipeline(df)
    joblib.dump(pipe, models.MODEL_PATH)

    snapshot, meta = build_hex_snapshot(df)
    snapshot.to_csv(HEX_SNAPSHOT_PATH, index=False)
    SNAPSHOT_META_PATH.write_text(json.dumps(meta))
    print(f"saved model + hex snapshot as of {meta['reference_date']}")


def geocode_postcode(postcode: str) -> tuple[float, float]:
    normalized = _normalize_postcode(pd.Series([postcode])).iloc[0]
    match = load_onspd_lookup({normalized})
    if match.empty:
        raise ValueError(f"postcode {postcode!r} not found in onspd")
    return match.iloc[0]["lat"], match.iloc[0]["long"]


def predict_price(postcode: str, property_type: str, duration: str, old_new: str = "N") -> float:
    lat, long = geocode_postcode(postcode)
    row = add_distance_to_centre(pd.DataFrame({"lat": [lat], "long": [long]}))
    if row.empty:
        raise ValueError(f"postcode {postcode!r} is outside the trained region (too far from central london)")
    row = add_hex_id(row)
    hex_id = row["hex_id"].iloc[0]

    snapshot = pd.read_csv(HEX_SNAPSHOT_PATH, dtype={"hex_id": "string"}).set_index("hex_id")
    meta = json.loads(SNAPSHOT_META_PATH.read_text())

    if hex_id in snapshot.index:
        hex_trailing_price = snapshot.loc[hex_id, "hex_trailing_price"]
        hex_trailing_count = snapshot.loc[hex_id, "hex_trailing_count"]
        hex_neighbor_lag = snapshot.loc[hex_id, "hex_neighbor_lag"]
    else:
        # cold-start hex, no recent sales at all -- same citywide fallback rule as training
        hex_trailing_price = meta["citywide_trailing_price"]
        hex_trailing_count = 0
        hex_neighbor_lag = meta["citywide_trailing_price"]

    features = pd.DataFrame([{
        "property_type": property_type,
        "old_new": old_new,
        "duration": duration,
        "dist_to_centre_km": row["dist_to_centre_km"].iloc[0],
        "sale_year": meta["reference_year"],
        "sale_month": meta["reference_month"],
        "hex_trailing_price": hex_trailing_price,
        "hex_trailing_count": hex_trailing_count,
        "hex_neighbor_lag": hex_neighbor_lag,
    }])

    pipe = joblib.load(models.MODEL_PATH)
    log_price = pipe.predict(features)[0]
    return 10 ** log_price


def main() -> None:
    parser = argparse.ArgumentParser(description="predict a house price for a postcode + property type + tenure")
    parser.add_argument("--train", action="store_true", help="fit the model and rebuild the hex snapshot")
    parser.add_argument("postcode", nargs="?")
    parser.add_argument("property_type", nargs="?", choices=["F", "S", "T", "D"])
    parser.add_argument("duration", nargs="?", choices=["F", "L"])
    parser.add_argument("--old-new", default="N", choices=["N", "Y"])
    args = parser.parse_args()

    if args.train:
        train_and_save()
        return

    price = predict_price(args.postcode, args.property_type, args.duration, args.old_new)
    print(f"predicted price: £{price:,.0f}")


if __name__ == "__main__":
    main()
