from __future__ import annotations

import argparse
import json

import h3
import joblib
import numpy as np
import pandas as pd

from src import models
from src.data_prep import DATA_PROCESSED, _normalize_postcode, load_onspd_lookup
from src.features import HEX_TARGET_ENCODING_K, TRAILING_WINDOW, add_distance_to_centre, add_hex_id, add_multi_res_hex_ids

HEX_SNAPSHOT_PATH = DATA_PROCESSED / "hex_snapshot.csv"
HEX9_SNAPSHOT_PATH = DATA_PROCESSED / "hex9_snapshot.csv"
SNAPSHOT_META_PATH = DATA_PROCESSED / "snapshot_meta.json"


def build_hex_snapshot(df: pd.DataFrame, reference_date) -> tuple[pd.DataFrame, dict]:
    # one fixed reference date instead of models.py's per-row rolling window --
    # this is what "current" local price looks like as of the end of the training data,
    # never real wall-clock today (the model never saw dates past this)
    window_start = reference_date - pd.Timedelta(TRAILING_WINDOW)
    recent = df[(df["date_of_transfer"] >= window_start) & (df["date_of_transfer"] < reference_date)]

    snapshot = recent.groupby("hex_id")["price"].agg(hex_trailing_price="median", hex_trailing_count="count")
    citywide_price = recent["price"].median()

    lookup = snapshot["hex_trailing_price"]

    def neighbour_avg(h):
        neighbours = [n for n in h3.grid_disk(h, 1) if n != h]
        vals = lookup.reindex(neighbours).dropna()
        return vals.mean() if len(vals) else lookup[h]

    snapshot["hex_neighbour_lag"] = [neighbour_avg(h) for h in snapshot.index]

    # target encoding is a single snapshot value per hex too -- same expanding-window +
    # shrinkage logic as add_hex_target_encoding, anchored at the fixed reference date
    prior = df[df["date_of_transfer"] < reference_date].copy()
    prior["_log_price"] = np.log10(prior["price"])
    citywide_log_price_mean = prior["_log_price"].mean()
    target_enc = prior.groupby("hex_id")["_log_price"].agg(["mean", "count"])
    target_enc["hex_target_encoded"] = (
        target_enc["count"] * target_enc["mean"] + HEX_TARGET_ENCODING_K * citywide_log_price_mean
    ) / (target_enc["count"] + HEX_TARGET_ENCODING_K)
    snapshot = snapshot.join(target_enc[["hex_target_encoded"]], how="left")
    # cold-start hex for target encoding (no prior sales at all): pure citywide mean, n=0
    snapshot["hex_target_encoded"] = snapshot["hex_target_encoded"].fillna(citywide_log_price_mean)

    meta = {
        "reference_date": str(reference_date.date()),
        "reference_year": int(reference_date.year),
        "reference_month": int(reference_date.month),
        "citywide_trailing_price": float(citywide_price),
        "citywide_log_price_mean": float(citywide_log_price_mean),
    }
    return snapshot.reset_index(), meta


def build_hex9_snapshot(df: pd.DataFrame, reference_date) -> pd.DataFrame:
    window_start = reference_date - pd.Timedelta(TRAILING_WINDOW)
    recent = df[(df["date_of_transfer"] >= window_start) & (df["date_of_transfer"] < reference_date)]
    return recent.groupby("hex9_id")["price"].median().rename("hex9_trailing_price").reset_index()


def train_and_save() -> None:
    df = models.load_dataset()
    pipe = models.fit_full_pipeline(df)
    joblib.dump(pipe, models.MODEL_PATH)

    lower_pipe, upper_pipe = models.fit_full_quantile_pipelines(df)
    joblib.dump(lower_pipe, models.QUANTILE_LOWER_PATH)
    joblib.dump(upper_pipe, models.QUANTILE_UPPER_PATH)

    reference_date = df["date_of_transfer"].max()
    snapshot, meta = build_hex_snapshot(df, reference_date)
    snapshot.to_csv(HEX_SNAPSHOT_PATH, index=False)
    build_hex9_snapshot(df, reference_date).to_csv(HEX9_SNAPSHOT_PATH, index=False)
    SNAPSHOT_META_PATH.write_text(json.dumps(meta))
    print(f"saved model + hex snapshots as of {meta['reference_date']}")


def geocode_postcode(postcode: str) -> tuple[float, float]:
    normalized = _normalize_postcode(pd.Series([postcode])).iloc[0]
    match = load_onspd_lookup({normalized})
    if match.empty:
        raise ValueError(f"postcode {postcode!r} not found in onspd")
    return match.iloc[0]["lat"], match.iloc[0]["long"]


def predict_price(postcode: str, property_type: str, duration: str, old_new: str = "N") -> tuple[float, float, float]:
    lat, long = geocode_postcode(postcode)
    row = add_distance_to_centre(pd.DataFrame({"lat": [lat], "long": [long]}))
    if row.empty:
        raise ValueError(f"postcode {postcode!r} is outside the trained region (too far from central london)")
    row = add_hex_id(row)
    row = add_multi_res_hex_ids(row, resolutions=(9,))
    hex_id, hex9_id = row["hex_id"].iloc[0], row["hex9_id"].iloc[0]

    snapshot = pd.read_csv(HEX_SNAPSHOT_PATH, dtype={"hex_id": "string"}).set_index("hex_id")
    hex9_snapshot = pd.read_csv(HEX9_SNAPSHOT_PATH, dtype={"hex9_id": "string"}).set_index("hex9_id")
    meta = json.loads(SNAPSHOT_META_PATH.read_text())

    if hex_id in snapshot.index:
        hex_trailing_price = snapshot.loc[hex_id, "hex_trailing_price"]
        hex_trailing_count = snapshot.loc[hex_id, "hex_trailing_count"]
        hex_neighbour_lag = snapshot.loc[hex_id, "hex_neighbour_lag"]
        hex_target_encoded = snapshot.loc[hex_id, "hex_target_encoded"]
    else:
        # cold-start hex, no recent sales at all -- same citywide fallback rule as training
        hex_trailing_price = meta["citywide_trailing_price"]
        hex_trailing_count = 0
        hex_neighbour_lag = meta["citywide_trailing_price"]
        hex_target_encoded = meta["citywide_log_price_mean"]

    if hex9_id in hex9_snapshot.index:
        hex9_trailing_price = hex9_snapshot.loc[hex9_id, "hex9_trailing_price"]
    else:
        hex9_trailing_price = meta["citywide_trailing_price"]

    ref_month = meta["reference_month"]
    features = pd.DataFrame([{
        "property_type": property_type,
        "old_new": old_new,
        "duration": duration,
        "dist_to_centre_km": row["dist_to_centre_km"].iloc[0],
        "sale_year": meta["reference_year"],
        "sale_month_sin": np.sin(2 * np.pi * ref_month / 12),
        "sale_month_cos": np.cos(2 * np.pi * ref_month / 12),
        "sdlt_rush": 0,  # reference date is always outside the mar/apr 2025 rush window
        "hex_trailing_price": hex_trailing_price,
        "hex_trailing_count": hex_trailing_count,
        "hex_neighbour_lag": hex_neighbour_lag,
        "hex9_trailing_price": hex9_trailing_price,
        "hex_target_encoded": hex_target_encoded,
    }])

    pipe = joblib.load(models.MODEL_PATH)
    lower_pipe = joblib.load(models.QUANTILE_LOWER_PATH)
    upper_pipe = joblib.load(models.QUANTILE_UPPER_PATH)

    point = 10 ** pipe.predict(features)[0]
    lower = 10 ** lower_pipe.predict(features)[0]
    upper = 10 ** upper_pipe.predict(features)[0]
    # independently-trained quantile models have no monotonicity guarantee against each
    # other or the point estimate -- guard rather than return a nonsensical crossed interval
    lower, upper = min(lower, point), max(upper, point)
    return point, lower, upper


def main() -> None:
    parser = argparse.ArgumentParser(description="predict a house price for a postcode + property type + tenure")
    parser.add_argument("--train", action="store_true", help="fit the model and rebuild the hex snapshots")
    parser.add_argument("postcode", nargs="?")
    parser.add_argument("property_type", nargs="?", choices=["F", "S", "T", "D"])
    parser.add_argument("duration", nargs="?", choices=["F", "L"])
    parser.add_argument("--old-new", default="N", choices=["N", "Y"])
    args = parser.parse_args()

    if args.train:
        train_and_save()
        return

    point, lower, upper = predict_price(args.postcode, args.property_type, args.duration, args.old_new)
    print(f"predicted price: £{point:,.0f} (80% interval: £{lower:,.0f} - £{upper:,.0f})")


if __name__ == "__main__":
    main()
