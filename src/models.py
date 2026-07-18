from __future__ import annotations

import numpy as np
import pandas as pd
from pygam import LinearGAM, f, l, s
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

from src.data_prep import DATA_PROCESSED, filter_residential, trim_price_outliers
from src.features import build_features
from src.spatial_diagnostics import morans_i_on_residuals, spatial_group_kfold

CATEGORICAL = ["property_type", "old_new", "duration"]
BASELINE_NUMERIC = ["dist_to_centre_km", "sale_year", "sale_month_sin", "sale_month_cos", "sdlt_rush"]
HEX_NUMERIC = ["hex_trailing_price", "hex_trailing_count", "hex_neighbor_lag"]
# needs a specific street/house, not just a postcode -- research/evaluation only for now,
# not wired into fit_full_pipeline/predict.py until the final feature set is settled
ADDRESS_NUMERIC = ["paon_is_named", "street_trailing_price"]
# accumulates through the improvement-batch phases -- compare_models/shap_summary track
# "how good can the model get", fit_full_pipeline stays on the serving-safe subset
RESEARCH_NUMERIC = BASELINE_NUMERIC + HEX_NUMERIC + ADDRESS_NUMERIC
TARGET = "price"
MODEL_PATH = DATA_PROCESSED / "xgb_pipeline.joblib"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / "london_house_prices.csv", parse_dates=["date_of_transfer"])
    df = filter_residential(df)
    df = trim_price_outliers(df)
    return build_features(df)


def make_frame(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    X = df[CATEGORICAL + numeric_cols].copy()
    y = np.log10(df[TARGET])
    return X, y


def linear_pipeline(numeric_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), numeric_cols),
    ])
    return Pipeline([("pre", pre), ("model", LinearRegression())])


def xgb_pipeline(numeric_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", numeric_cols),
    ])
    model = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
    return Pipeline([("pre", pre), ("model", model)])


def gam_pipeline(numeric_cols: list[str]) -> Pipeline:
    # gam wants factors as ordinal-coded ints (f terms), not one-hot -- one-hot would
    # give each dummy its own smoothing penalty instead of a single per-feature term
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
        ("num", StandardScaler(), numeric_cols),
    ])
    n_cat = len(CATEGORICAL)
    # smooth terms on distance/hex location features, factor terms on property attrs,
    # coarser smoothing on sale_year (not spatial, no reason to expect wiggle), plain
    # linear terms for the sin/cos pair and the binary sdlt flag -- a sinusoid pair is
    # already the smooth shape, and a 0/1 flag has nothing to smooth
    linear_cols = ("sale_month_sin", "sale_month_cos", "sdlt_rush", "paon_is_named")
    terms = None
    for i in range(n_cat):
        terms = f(i) if terms is None else terms + f(i)
    for j, col in enumerate(numeric_cols):
        idx = n_cat + j
        if col == "sale_year":
            term = s(idx, n_splines=6)
        elif col in linear_cols:
            term = l(idx)
        else:
            term = s(idx)
        terms = terms + term
    gam = LinearGAM(terms)
    return Pipeline([("pre", pre), ("model", gam)])


def fit_full_pipeline(df: pd.DataFrame) -> Pipeline:
    numeric_cols = BASELINE_NUMERIC + HEX_NUMERIC
    X, y = make_frame(df, numeric_cols)
    pipe = xgb_pipeline(numeric_cols)
    pipe.fit(X, y)
    return pipe


def evaluate_cv(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, pipeline_fn, numeric_cols: list[str]) -> dict:
    rmses, maes, r2s = [], [], []
    for train_idx, test_idx in spatial_group_kfold(df):
        pipe = pipeline_fn(numeric_cols)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        rmses.append(root_mean_squared_error(y.iloc[test_idx], pred))
        maes.append(mean_absolute_error(y.iloc[test_idx], pred))
        r2s.append(r2_score(y.iloc[test_idx], pred))
    return {"rmse": np.mean(rmses), "mae": np.mean(maes), "r2": np.mean(r2s)}


def out_of_fold_residuals(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, numeric_cols: list[str]) -> pd.Series:
    # residuals must come from held-out folds -- in-sample residuals from a fitted model
    # understate autocorrelation because the model has already absorbed local structure
    resid = pd.Series(index=X.index, dtype=float)
    for train_idx, test_idx in spatial_group_kfold(df):
        pipe = linear_pipeline(numeric_cols)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        resid.iloc[test_idx] = y.iloc[test_idx].values - pred
    return resid


def compare_models(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = RESEARCH_NUMERIC
    X, y = make_frame(df, numeric_cols)
    results = {
        "linear": evaluate_cv(df, X, y, linear_pipeline, numeric_cols),
        "gam": evaluate_cv(df, X, y, gam_pipeline, numeric_cols),
        "xgboost": evaluate_cv(df, X, y, xgb_pipeline, numeric_cols),
    }
    return pd.DataFrame(results).T


def moran_before_after(df: pd.DataFrame) -> pd.DataFrame:
    X_base, y = make_frame(df, BASELINE_NUMERIC)
    X_hex, _ = make_frame(df, BASELINE_NUMERIC + HEX_NUMERIC)
    resid_base = out_of_fold_residuals(df, X_base, y, BASELINE_NUMERIC)
    resid_hex = out_of_fold_residuals(df, X_hex, y, BASELINE_NUMERIC + HEX_NUMERIC)

    m_base = morans_i_on_residuals(df.assign(resid=resid_base), "resid")
    m_hex = morans_i_on_residuals(df.assign(resid=resid_hex), "resid")
    return pd.DataFrame(
        {"moran_i": [m_base.I, m_hex.I], "p_sim": [m_base.p_sim, m_hex.p_sim]},
        index=["baseline (no hex features)", "hex features"],
    )


def shap_summary(df: pd.DataFrame, sample_size: int = 5000):
    import shap

    numeric_cols = RESEARCH_NUMERIC
    X, y = make_frame(df, numeric_cols)
    pipe = xgb_pipeline(numeric_cols)
    pipe.fit(X, y)

    feature_names = pipe.named_steps["pre"].get_feature_names_out()
    X_transformed = pipe.named_steps["pre"].transform(X)
    sample_idx = np.random.RandomState(0).choice(len(X_transformed), size=min(sample_size, len(X_transformed)), replace=False)

    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    shap_values = explainer.shap_values(X_transformed[sample_idx])
    return shap.Explanation(shap_values, feature_names=feature_names, data=X_transformed[sample_idx])


def main() -> None:
    df = load_dataset()
    print(f"loaded {len(df):,} rows")

    print("\nmodel comparison (5-fold spatial block cv, log10 price)")
    print(compare_models(df))

    print("\nmoran's i on cv residuals, baseline vs hex features")
    print(moran_before_after(df))


if __name__ == "__main__":
    main()
