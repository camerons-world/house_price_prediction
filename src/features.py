from __future__ import annotations

import numpy as np
import pandas as pd
import h3

# res 8: median ~85 sales/hex over the window, 90th pct of hexes have >=10 -- res 9
# drops to single digits for many hexes, res 7 is too coarse (median 520/hex)
HEX_RESOLUTION = 8
TRAILING_WINDOW = "365D"
LONDON_CENTRE = (51.5074, -0.1278)  # charing cross, conventional london zero point


def add_hex_id(df: pd.DataFrame) -> pd.DataFrame:
    df["hex_id"] = [h3.latlng_to_cell(lat, lon, HEX_RESOLUTION) for lat, lon in zip(df["lat"], df["long"])]
    return df


MAX_RADIUS_KM = 32  # greater london's furthest genuine points (havering/bromley) sit ~30-32km out


def add_distance_to_centre(df: pd.DataFrame) -> pd.DataFrame:
    lat1, lon1 = np.radians(df["lat"]), np.radians(df["long"])
    lat2, lon2 = np.radians(LONDON_CENTRE[0]), np.radians(LONDON_CENTRE[1])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    df["dist_to_centre_km"] = 2 * 6371 * np.arcsin(np.sqrt(a))
    # catches rare postcode/geocoding errors in the source data (e.g. a southend
    # postcode on an otherwise-genuine newham record) -- not a modeling choice
    return df[df["dist_to_centre_km"] <= MAX_RADIUS_KM].reset_index(drop=True)


def add_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    df["sale_year"] = df["date_of_transfer"].dt.year
    df["sale_month"] = df["date_of_transfer"].dt.month
    return df


def add_hex_trailing_stats(df: pd.DataFrame) -> pd.DataFrame:
    # trailing = strictly-prior sales only (closed='left' excludes same-day ties too) --
    # this is the leakage rule from PLAN.md, not optional
    df = df.sort_values("date_of_transfer").reset_index(drop=True)

    rolled = (
        df.set_index("date_of_transfer")
        .groupby("hex_id")["price"]
        .rolling(TRAILING_WINDOW, closed="left")
        .agg(["median", "count"])
        .reset_index()
        .drop_duplicates(subset=["hex_id", "date_of_transfer"])
        .rename(columns={"median": "hex_trailing_price", "count": "hex_trailing_count"})
    )
    df = df.merge(rolled, on=["hex_id", "date_of_transfer"], how="left")

    # cold-start hex (no prior sale in-window): price falls back to citywide median,
    # count is genuinely 0 -- citywide count is a different scale, not a valid fallback
    citywide_price = df.set_index("date_of_transfer")["price"].rolling(TRAILING_WINDOW, closed="left").median()
    citywide_price.index = df.index
    df["hex_trailing_price"] = df["hex_trailing_price"].fillna(citywide_price)
    df["hex_trailing_count"] = df["hex_trailing_count"].fillna(0)

    # handful of sales on day 1 of the window have no prior data at all, citywide or
    # otherwise -- genuine cold start, nothing to fall back to, so drop rather than fabricate
    df = df.dropna(subset=["hex_trailing_price"]).reset_index(drop=True)
    return df, rolled


def add_neighbor_lag(df: pd.DataFrame, trailing_lookup: pd.DataFrame) -> pd.DataFrame:
    # k=1 ring average of each neighbouring hex's own trailing price, as of <= this row's
    # date (merge_asof direction='backward') -- same strictly-prior rule, extended to neighbours
    neighbor_lookup = trailing_lookup[["hex_id", "date_of_transfer", "hex_trailing_price"]].rename(
        columns={"hex_id": "neighbor_hex", "hex_trailing_price": "neighbor_price"}
    ).sort_values("date_of_transfer")

    df["_row_id"] = df.index
    neighbor_ids = df["hex_id"].apply(lambda h: [n for n in h3.grid_disk(h, 1) if n != h])
    exploded = df[["_row_id", "date_of_transfer"]].join(neighbor_ids.rename("neighbor_hex"))
    exploded = exploded.explode("neighbor_hex").sort_values("date_of_transfer")

    merged = pd.merge_asof(exploded, neighbor_lookup, on="date_of_transfer", by="neighbor_hex", direction="backward")
    lag = merged.groupby("_row_id")["neighbor_price"].mean()
    df["hex_neighbor_lag"] = df["_row_id"].map(lag)
    df["hex_neighbor_lag"] = df["hex_neighbor_lag"].fillna(df["hex_trailing_price"])
    return df.drop(columns="_row_id")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_hex_id(df.copy())
    df = add_distance_to_centre(df)
    df = add_date_parts(df)
    df, trailing_lookup = add_hex_trailing_stats(df)
    df = add_neighbor_lag(df, trailing_lookup)
    return df
