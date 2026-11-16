import pandas as pd

from src.data_prep import _normalize_postcode, filter_residential, trim_price_outliers


def test_normalize_postcode_strips_uppercases_and_collapses_whitespace():
    out = _normalize_postcode(pd.Series(["  sw1a   1aa ", "e1\t6an", "N1 9GU"]))
    assert out.tolist() == ["SW1A 1AA", "E1 6AN", "N1 9GU"]


def test_filter_residential_drops_category_b_and_type_o():
    df = pd.DataFrame({
        "ppd_category_type": ["A", "B", "A", "A", "B"],
        "property_type": ["D", "D", "O", "T", "O"],
        "price": [1, 2, 3, 4, 5],
    })
    out = filter_residential(df)
    assert out["price"].tolist() == [1, 4]
    assert out.index.tolist() == [0, 1]


def test_trim_price_outliers_drops_below_floor_and_above_cap():
    df = pd.DataFrame({"price": [1, 9_999, *range(100_000, 100_000 + 998), 500_000_000]})
    out = trim_price_outliers(df)
    assert out["price"].min() == 100_000
    assert out["price"].max() < 500_000_000
    assert 1 not in out["price"].values and 9_999 not in out["price"].values


def test_trim_price_outliers_keeps_price_at_floor():
    df = pd.DataFrame({"price": [10_000, 200_000, 300_000]})
    out = trim_price_outliers(df, cap_quantile=1.0)
    assert out["price"].tolist() == [10_000, 200_000, 300_000]
