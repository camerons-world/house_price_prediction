from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from pygam import LinearGAM, f, l, s
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

from src.data_prep import DATA_PROCESSED, filter_residential, trim_price_outliers
from src.features import build_features
from src.spatial_diagnostics import morans_i_on_residuals, spatial_group_kfold

CATEGORICAL = ["property_type", "old_new", "duration"]
BASELINE_NUMERIC = [
    "dist_to_centre_km",
    "sale_year",
    "sale_month_sin",
    "sale_month_cos",
    "sdlt_rush",
]
HEX_NUMERIC = ["hex_trailing_price", "hex_trailing_count", "hex_neighbour_lag"]
# needs a specific street/house, not just a postcode -- research/evaluation only for now,
# not wired into fit_full_pipeline/predict.py until the final feature set is settled
ADDRESS_NUMERIC = ["paon_is_named", "street_trailing_price"]
# res 9 (finer local trend) measured as a standalone win via evaluate_cv ablation --
# res 7 and a k=2 neighbour ring were also tried and dropped, see features.py
MULTI_RES_NUMERIC = ["hex9_trailing_price"]
# expanding-window, shrinkage-smoothed hex price level -- correlation with hex_trailing_price
# is 0.81 (not redundant) with a real r2 lift (+0.0024), passes the ablation gate
TARGET_ENC_NUMERIC = ["hex_target_encoded"]
# accumulates through the improvement-batch phases -- compare_models/shap_summary track
# "how good can the model get". hex9/target-encoding only need a postcode (hex-level,
# precomputable), so they're servable -- ADDRESS_NUMERIC needs a specific street/house a
# postcode-only query doesn't have, so it stays research-only, excluded from serving
RESEARCH_NUMERIC = (
    BASELINE_NUMERIC + HEX_NUMERIC + ADDRESS_NUMERIC + MULTI_RES_NUMERIC + TARGET_ENC_NUMERIC
)
SERVING_NUMERIC = BASELINE_NUMERIC + HEX_NUMERIC + MULTI_RES_NUMERIC + TARGET_ENC_NUMERIC
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
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), numeric_cols),
        ]
    )
    return Pipeline([("pre", pre), ("model", LinearRegression())])


def xgb_pipeline(numeric_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", numeric_cols),
        ]
    )
    # tuned via tune_xgb's spatial-cv randomized search -- marginal over the original
    # hand-picked values (rmse 0.1475 -> 0.1474), adopted anyway since it cost nothing
    model = XGBRegressor(
        n_estimators=300, max_depth=7, learning_rate=0.03, subsample=0.8, colsample_bytree=1.0
    )
    return Pipeline([("pre", pre), ("model", model)])


def gam_pipeline(numeric_cols: list[str]) -> Pipeline:
    # gam wants factors as ordinal-coded ints (f terms), not one-hot -- one-hot would
    # give each dummy its own smoothing penalty instead of a single per-feature term
    pre = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL,
            ),
            ("num", StandardScaler(), numeric_cols),
        ]
    )
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


def lgbm_pipeline(numeric_cols: list[str]) -> Pipeline:
    # ordinal-encoded + passthrough, same simplification as gam_pipeline -- lightgbm's
    # "native" categorical splits need category-dtype columns preserved through the
    # pipeline, which ColumnTransformer's default array output doesn't keep. this forfeits
    # that specific advantage but is still a fair, working comparison against xgboost
    pre = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL,
            ),
            ("num", "passthrough", numeric_cols),
        ]
    )
    model = lgb.LGBMRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=-1,
    )
    return Pipeline([("pre", pre), ("model", model)])


def tune_xgb(
    df: pd.DataFrame, numeric_cols: list[str] = RESEARCH_NUMERIC, n_iter: int = 20
) -> tuple[dict, float]:
    # xgb_pipeline's hyperparameters were hand-picked, never searched -- spatial_group_kfold
    # as the cv splitter here too, a plain k-fold search would leak the same way plain
    # k-fold evaluation would
    X, y = make_frame(df, numeric_cols)
    param_dist = {
        "model__max_depth": [3, 4, 5, 6, 7, 8],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "model__n_estimators": [200, 300, 400, 600],
        "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    }
    cv = list(spatial_group_kfold(df, n_splits=5))
    search = RandomizedSearchCV(
        xgb_pipeline(numeric_cols),
        param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=0,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_params_, -search.best_score_


QUANTILE_ALPHA_LOWER, QUANTILE_ALPHA_UPPER = 0.1, 0.9  # 80% interval
QUANTILE_LOWER_PATH = DATA_PROCESSED / "xgb_quantile_lower.joblib"
QUANTILE_UPPER_PATH = DATA_PROCESSED / "xgb_quantile_upper.joblib"


def xgb_quantile_pipeline(numeric_cols: list[str], alpha: float) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", numeric_cols),
        ]
    )
    # native xgboost quantile regression -- two independent single-alpha models rather
    # than one multi-quantile model, and rather than adding a mapie dependency for
    # conformal calibration we don't need yet (see evaluate_interval_cv's coverage check)
    model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        n_estimators=300,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=1.0,
    )
    return Pipeline([("pre", pre), ("model", model)])


def fit_full_quantile_pipelines(df: pd.DataFrame) -> tuple[Pipeline, Pipeline]:
    numeric_cols = SERVING_NUMERIC
    X, y = make_frame(df, numeric_cols)
    lower = xgb_quantile_pipeline(numeric_cols, QUANTILE_ALPHA_LOWER).fit(X, y)
    upper = xgb_quantile_pipeline(numeric_cols, QUANTILE_ALPHA_UPPER).fit(X, y)
    return lower, upper


def evaluate_interval_cv(
    df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, numeric_cols: list[str]
) -> dict:
    # coverage alone is gameable by a too-wide interval, width alone says nothing about
    # correctness -- report both, computed the same spatial-cv way as everything else
    coverages, widths = [], []
    for train_idx, test_idx in spatial_group_kfold(df):
        lo_pipe = xgb_quantile_pipeline(numeric_cols, QUANTILE_ALPHA_LOWER).fit(
            X.iloc[train_idx], y.iloc[train_idx]
        )
        hi_pipe = xgb_quantile_pipeline(numeric_cols, QUANTILE_ALPHA_UPPER).fit(
            X.iloc[train_idx], y.iloc[train_idx]
        )
        lo, hi = lo_pipe.predict(X.iloc[test_idx]), hi_pipe.predict(X.iloc[test_idx])
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        y_true = y.iloc[test_idx].values
        coverages.append(((y_true >= lo) & (y_true <= hi)).mean())
        widths.append((hi - lo).mean())
    return {
        "empirical_coverage": np.mean(coverages),
        "target_coverage": QUANTILE_ALPHA_UPPER - QUANTILE_ALPHA_LOWER,
        "mean_interval_width_log10": np.mean(widths),
    }


def fit_full_pipeline(df: pd.DataFrame) -> Pipeline:
    numeric_cols = SERVING_NUMERIC
    X, y = make_frame(df, numeric_cols)
    pipe = xgb_pipeline(numeric_cols)
    pipe.fit(X, y)
    return pipe


def evaluate_cv(
    df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, pipeline_fn, numeric_cols: list[str]
) -> dict:
    rmses, maes, r2s = [], [], []
    for train_idx, test_idx in spatial_group_kfold(df):
        pipe = pipeline_fn(numeric_cols)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        rmses.append(root_mean_squared_error(y.iloc[test_idx], pred))
        maes.append(mean_absolute_error(y.iloc[test_idx], pred))
        r2s.append(r2_score(y.iloc[test_idx], pred))
    return {"rmse": np.mean(rmses), "mae": np.mean(maes), "r2": np.mean(r2s)}


def out_of_fold_residuals(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list[str],
    pipeline_fn=linear_pipeline,
) -> pd.Series:
    # residuals must come from held-out folds -- in-sample residuals from a fitted model
    # understate autocorrelation because the model has already absorbed local structure.
    # defaults to linear (moran_before_after's fixed diagnostic model), but any pipeline_fn
    # works -- e.g. xgb_pipeline, to check the actual best model's residual autocorrelation
    resid = pd.Series(index=X.index, dtype=float)
    for train_idx, test_idx in spatial_group_kfold(df):
        pipe = pipeline_fn(numeric_cols)
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
        "lightgbm": evaluate_cv(df, X, y, lgbm_pipeline, numeric_cols),
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
    sample_idx = np.random.RandomState(0).choice(
        len(X_transformed), size=min(sample_size, len(X_transformed)), replace=False
    )

    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    shap_values = explainer.shap_values(X_transformed[sample_idx])
    return shap.Explanation(
        shap_values, feature_names=feature_names, data=X_transformed[sample_idx]
    )


def main() -> None:
    df = load_dataset()
    print(f"loaded {len(df):,} rows")

    print("\nmodel comparison (5-fold spatial block cv, log10 price)")
    print(compare_models(df))

    print("\nmoran's i on cv residuals, baseline vs hex features")
    print(moran_before_after(df))


if __name__ == "__main__":
    main()
