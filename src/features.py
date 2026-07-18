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


SDLT_RUSH_WINDOW = ("2025-03-01", "2025-04-30")  # sdlt nil-rate band reverted 1 apr 2025 -- see eda notebook


def add_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    df["sale_year"] = df["date_of_transfer"].dt.year
    month = df["date_of_transfer"].dt.month
    # cyclical encoding -- dec/jan are numerically 11 apart but 1 month apart in reality
    df["sale_month_sin"] = np.sin(2 * np.pi * month / 12)
    df["sale_month_cos"] = np.cos(2 * np.pi * month / 12)
    df["sdlt_rush"] = df["date_of_transfer"].between(*SDLT_RUSH_WINDOW).astype(int)
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


def add_neighbor_lag(df: pd.DataFrame, trailing_lookup: pd.DataFrame, k: int = 1, col_name: str = "hex_neighbor_lag") -> pd.DataFrame:
    # k-ring average of each neighbouring hex's own trailing price, as of <= this row's
    # date (merge_asof direction='backward') -- same strictly-prior rule, extended to neighbours
    neighbor_lookup = trailing_lookup[["hex_id", "date_of_transfer", "hex_trailing_price"]].rename(
        columns={"hex_id": "neighbor_hex", "hex_trailing_price": "neighbor_price"}
    ).sort_values("date_of_transfer")

    df["_row_id"] = df.index
    neighbor_ids = df["hex_id"].apply(lambda h: [n for n in h3.grid_disk(h, k) if n != h])
    exploded = df[["_row_id", "date_of_transfer"]].join(neighbor_ids.rename("neighbor_hex"))
    exploded = exploded.explode("neighbor_hex").sort_values("date_of_transfer")

    merged = pd.merge_asof(exploded, neighbor_lookup, on="date_of_transfer", by="neighbor_hex", direction="backward")
    lag = merged.groupby("_row_id")["neighbor_price"].mean()
    df[col_name] = df["_row_id"].map(lag)
    df[col_name] = df[col_name].fillna(df["hex_trailing_price"])
    return df.drop(columns="_row_id")


MULTI_RES_RESOLUTIONS = (9,)  # alongside the primary res 8 -- finer local trend. res 7 (coarser)
# and a k=2 neighbor ring were also tried and measured (evaluate_cv ablation) but added
# nothing once res 9 was in, so they were dropped rather than carried as unused features


def add_multi_res_hex_ids(df: pd.DataFrame, resolutions=MULTI_RES_RESOLUTIONS) -> pd.DataFrame:
    for res in resolutions:
        df[f"hex{res}_id"] = [h3.latlng_to_cell(lat, lon, res) for lat, lon in zip(df["lat"], df["long"])]
    return df


def add_multi_res_trailing_price(df: pd.DataFrame, resolutions=MULTI_RES_RESOLUTIONS) -> pd.DataFrame:
    # same trailing-window, strictly-prior discipline as add_hex_trailing_stats, just at
    # coarser/finer hex sizes -- res 8 already carries the primary price/count/neighbor-lag
    citywide_price = df.set_index("date_of_transfer")["price"].rolling(TRAILING_WINDOW, closed="left").median()
    citywide_price.index = df.index

    for res in resolutions:
        hex_col, price_col = f"hex{res}_id", f"hex{res}_trailing_price"
        rolled = (
            df.set_index("date_of_transfer")
            .groupby(hex_col)["price"]
            .rolling(TRAILING_WINDOW, closed="left")
            .median()
            .reset_index()
            .drop_duplicates(subset=[hex_col, "date_of_transfer"])
            .rename(columns={"price": price_col})
        )
        df = df.merge(rolled, on=[hex_col, "date_of_transfer"], how="left")
        df[price_col] = df[price_col].fillna(citywide_price)

    # dropping res-8 cold-start rows upstream can expose a fresh "first ever row" with no
    # citywide history of its own -- same genuine-cold-start rule, drop rather than fabricate
    price_cols = [f"hex{res}_trailing_price" for res in resolutions]
    return df.dropna(subset=price_cols).reset_index(drop=True)


def add_paon_named_flag(df: pd.DataFrame) -> pd.DataFrame:
    # a named property ("THE OLD RECTORY") vs a plain house number is a free signal
    # correlated with age/prestige, sitting unused in a column no feature touched before
    df["paon_is_named"] = df["paon"].str.contains(r"[A-Za-z]{3,}", regex=True, na=False).astype(int)
    return df


STREET_SHRINKAGE_K = 8  # eda notebook: median 4 sales/street over the whole window -- thin
# streets should lean hard on the hex price, not be trusted on a handful of their own sales


def add_street_shrunk_price(df: pd.DataFrame) -> pd.DataFrame:
    # street names repeat across boroughs ("HIGH STREET" everywhere) -- must key on
    # street+district together or unrelated streets silently get merged into one series
    df = df.sort_values("date_of_transfer").reset_index(drop=True)
    df["_street_key"] = df["street"].astype(str) + "||" + df["district"].astype(str)

    # expanding (all strictly-prior history), not the 365D hex window -- a street sees a
    # handful of sales across the whole 3-year span, a 1-year window leaves nothing to use
    span_days = (df["date_of_transfer"].max() - df["date_of_transfer"].min()).days + 1
    expanding_window = f"{span_days}D"

    rolled = (
        df.set_index("date_of_transfer")
        .groupby("_street_key")["price"]
        .rolling(expanding_window, closed="left")
        .agg(["mean", "count"])
        .reset_index()
        .drop_duplicates(subset=["_street_key", "date_of_transfer"])
        .rename(columns={"mean": "_street_mean", "count": "_street_count"})
    )
    df = df.merge(rolled, on=["_street_key", "date_of_transfer"], how="left")

    # cold-start street (no prior sale on it at all): full weight on the hex price
    df["_street_mean"] = df["_street_mean"].fillna(df["hex_trailing_price"])
    df["_street_count"] = df["_street_count"].fillna(0)
    df["street_trailing_price"] = (
        df["_street_count"] * df["_street_mean"] + STREET_SHRINKAGE_K * df["hex_trailing_price"]
    ) / (df["_street_count"] + STREET_SHRINKAGE_K)

    return df.drop(columns=["_street_key", "_street_mean", "_street_count"])


HEX_TARGET_ENCODING_K = 20.0  # smoothing constant -- thin hexes lean harder on the citywide mean


def add_hex_target_encoding(df: pd.DataFrame, smoothing_k: float = HEX_TARGET_ENCODING_K) -> pd.DataFrame:
    # deliberately different from hex_trailing_price: expanding (all strictly-prior history,
    # not a 365D window) captures long-run price tier rather than recent momentum, and
    # count-weighted shrinkage toward the citywide mean denoises thin hexes instead of
    # hex_trailing_price's hard cutover (raw stat regardless of n, or full citywide if n=0)
    df = df.sort_values("date_of_transfer").reset_index(drop=True)
    df["_log_price"] = np.log10(df["price"])
    span_days = (df["date_of_transfer"].max() - df["date_of_transfer"].min()).days + 1
    expanding_window = f"{span_days}D"

    rolled = (
        df.set_index("date_of_transfer")
        .groupby("hex_id")["_log_price"]
        .rolling(expanding_window, closed="left")
        .agg(["mean", "count"])
        .reset_index()
        .drop_duplicates(subset=["hex_id", "date_of_transfer"])
        .rename(columns={"mean": "_hex_expanding_mean", "count": "_hex_expanding_count"})
    )
    df = df.merge(rolled, on=["hex_id", "date_of_transfer"], how="left")

    citywide_mean = df.set_index("date_of_transfer")["_log_price"].rolling(expanding_window, closed="left").mean()
    citywide_mean.index = df.index
    df["_hex_expanding_mean"] = df["_hex_expanding_mean"].fillna(citywide_mean)
    df["_hex_expanding_count"] = df["_hex_expanding_count"].fillna(0)
    df["hex_target_encoded"] = (
        df["_hex_expanding_count"] * df["_hex_expanding_mean"] + smoothing_k * citywide_mean
    ) / (df["_hex_expanding_count"] + smoothing_k)

    # same genuine-cold-start rule as everywhere else -- the very first row(s) have no
    # citywide history either, nothing to shrink toward, drop rather than fabricate
    df = df.dropna(subset=["hex_target_encoded"]).reset_index(drop=True)
    return df.drop(columns=["_log_price", "_hex_expanding_mean", "_hex_expanding_count"])


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_hex_id(df.copy())
    df = add_multi_res_hex_ids(df)
    df = add_distance_to_centre(df)
    df = add_date_parts(df)
    df, trailing_lookup = add_hex_trailing_stats(df)
    df = add_multi_res_trailing_price(df)
    df = add_neighbor_lag(df, trailing_lookup)
    df = add_paon_named_flag(df)
    df = add_street_shrunk_price(df)
    df = add_hex_target_encoding(df)
    return df
