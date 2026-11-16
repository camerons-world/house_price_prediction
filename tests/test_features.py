import pandas as pd
import pytest

from src.features import LONDON_CENTRE, add_distance_to_centre, add_hex_trailing_stats


def _sales(hex_id, dates, prices):
    return pd.DataFrame(
        {
            "hex_id": [hex_id] * len(dates),
            "date_of_transfer": pd.to_datetime(dates),
            "price": prices,
        }
    )


def test_trailing_price_excludes_same_day_and_future_sales():
    df = _sales(
        "h1",
        ["2023-01-01", "2023-01-05", "2023-01-10", "2023-01-10", "2023-06-01"],
        [100, 200, 999_999, 999_999, 300],
    )
    out, _ = add_hex_trailing_stats(df)
    last = out[out["date_of_transfer"] == "2023-06-01"].iloc[0]
    # same-day pair (jan 10) must not leak into a later trailing stat
    assert last["hex_trailing_price"] == pytest.approx(
        pd.Series([100, 200, 999_999, 999_999]).median()
    )

    jan10 = out[out["date_of_transfer"] == "2023-01-10"]
    # neither jan-10 row can see the other jan-10 sale (closed='left' excludes ties)
    assert (jan10["hex_trailing_price"] == 150).all()


def test_trailing_price_falls_outside_window():
    df = _sales("h1", ["2022-01-01", "2023-06-01", "2023-06-02"], [500, 10, 20])
    out, _ = add_hex_trailing_stats(df)
    # 2022-01-01 sale is >365d before the june sales, must not count towards their trailing stats
    june = out[out["date_of_transfer"] == "2023-06-02"].iloc[0]
    assert june["hex_trailing_price"] == 10


def test_cold_start_rows_are_dropped_not_fabricated():
    df = _sales("h1", ["2023-01-01"], [100])
    out, _ = add_hex_trailing_stats(df)
    assert out.empty


def test_distance_to_centre_zero_at_centre():
    df = pd.DataFrame({"lat": [LONDON_CENTRE[0]], "long": [LONDON_CENTRE[1]]})
    out = add_distance_to_centre(df)
    assert out["dist_to_centre_km"].iloc[0] == pytest.approx(0, abs=1e-6)


def test_distance_to_centre_drops_out_of_region_points():
    df = pd.DataFrame({"lat": [LONDON_CENTRE[0], 53.4808], "long": [LONDON_CENTRE[1], -2.2426]})
    out = add_distance_to_centre(df)
    # second point is manchester, ~260km out -- must be filtered, not just flagged
    assert len(out) == 1
