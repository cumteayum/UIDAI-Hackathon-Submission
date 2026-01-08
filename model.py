# Model
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import json
from datetime import datetime
import re
from difflib import get_close_matches


def canonicalize_text(s):
    if pd.isna(s):
        return s
    s = s.strip().lower()
    s = re.sub(r"[^a-z\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def harmonize_admin_names(df, col, similarity_threshold=0.9):
    """
    Canonicalize duplicate admin names using fuzzy matching.
    """
    uniques = sorted(df[col].dropna().unique())
    canonical_map = {}

    for name in uniques:
        if name in canonical_map:
            continue
        matches = get_close_matches(name, uniques, n=5, cutoff=similarity_threshold)
        for m in matches:
            canonical_map[m] = name

    df[col] = df[col].map(canonical_map)
    return df


LOG_PATH = "model_log.txt"
FORECAST_PATH = "forecast_2yr.csv"


def log_header(title):
    with open(LOG_PATH, "a") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(title + "\n")
        f.write("=" * 80 + "\n")


def log_text(text):
    with open(LOG_PATH, "a") as f:
        f.write(text + "\n")


def log_dict(d, indent=2):
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(d, indent=indent, default=str) + "\n")


def reset_logs():
    with open(LOG_PATH, "w") as f:
        f.write(f"MODEL LOG START — {datetime.now()}\n")


def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


CATBOOST_PARAMS = {
    "loss_function": "Poisson",
    "iterations": 600,
    "learning_rate": 0.05,
    "depth": 6,
    "eval_metric": "RMSE",
    "random_seed": 42,
    "early_stopping_rounds": 100,
}
FEATURES = [
    "state",
    "district",
    "pincode",
    "dow",
    "is_weekend",
    "month_sin",
    "month_cos",
    "lag_1",
    "lag_7",
    "rolling_7",
    "child_ratio",
    "adult_ratio",
]

CAT_FEATURES = ["state", "district", "pincode"]
log_header("MODEL CONFIGURATION")
log_dict(CATBOOST_PARAMS)
log_text(f"Features used: {FEATURES}")
log_text(f"Categorical features: {CAT_FEATURES}")


def add_global_fallbacks(df):
    global_mean = df["total_enrol"].mean()
    df["lag_1"] = df["lag_1"].fillna(global_mean)
    df["lag_7"] = df["lag_7"].fillna(global_mean)
    df["rolling_7"] = df["rolling_7"].fillna(global_mean)
    return df


def load_and_preprocess(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    df["state"] = df["state"].apply(canonicalize_text)
    df["district"] = df["district"].apply(canonicalize_text)

    df = harmonize_admin_names(df, "state")
    df = harmonize_admin_names(df, "district")

    df = df[df["age_0_5"] != "age_0_5"].reset_index(drop=True)
    df = df[df["age_5_17"] != "age_5_17"].reset_index(drop=True)
    df = df[df["age_18_greater"] != "age_18_greater"].reset_index(drop=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    age_cols = ["age_0_5", "age_5_17", "age_18_greater"]
    for col in age_cols:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ""), errors="coerce"
        )

    df[age_cols] = df[age_cols].fillna(df[age_cols].median())
    df["total_enrol"] = df[age_cols].sum(axis=1)

    df = df.sort_values(["state", "date"]).reset_index(drop=True)

    df["child_ratio"] = (df["age_0_5"] + df["age_5_17"]) / df["total_enrol"].replace(
        0, np.nan
    )
    df["adult_ratio"] = df["age_18_greater"] / df["total_enrol"].replace(0, np.nan)
    df[["child_ratio", "adult_ratio"]] = df[["child_ratio", "adult_ratio"]].fillna(0)

    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["lag_1"] = df.groupby("state")["total_enrol"].shift(1)
    df["lag_7"] = df.groupby("state")["total_enrol"].shift(7)
    df["rolling_7"] = df.groupby("state")["total_enrol"].shift(1).rolling(7).mean()

    log_header("DATASET SUMMARY")
    log_text(f"Rows: {len(df)}")
    log_text(f"Date range: {df['date'].min()} → {df['date'].max()}")
    log_text(f"States: {df['state'].nunique()}")
    log_text(f"Total enrolments: {df['total_enrol'].sum():,.0f}")
    log_text(f"Mean daily enrolment: {df['total_enrol'].mean():.2f}")

    df = df.dropna().reset_index(drop=True)
    return df


def train_and_evaluate(train_df, test_df, label):
    X_train = train_df[FEATURES]
    y_train = np.log1p(train_df["total_enrol"])

    X_test = test_df[FEATURES]
    y_test_raw = test_df["total_enrol"]
    y_test = np.log1p(y_test_raw)

    train_pool = Pool(X_train, y_train, cat_features=CAT_FEATURES)
    test_pool = Pool(X_test, y_test, cat_features=CAT_FEATURES)

    model = CatBoostRegressor(
        loss_function="Poisson",
        iterations=600,
        learning_rate=0.05,
        depth=6,
        eval_metric="RMSE",
        random_seed=42,
        early_stopping_rounds=100,
        verbose=100,
    )

    model.fit(train_pool, eval_set=test_pool)

    preds = np.expm1(model.predict(test_pool)).clip(min=0)

    rmse = np.sqrt(mean_squared_error(y_test_raw, preds))
    mae = mean_absolute_error(y_test_raw, preds)
    mape = safe_mape(y_test_raw, preds)

    print(f"\n===== {label} =====")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE : {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")

    # Error by state
    error_by_state = (
        test_df.assign(pred=preds)
        .groupby("state")
        .apply(lambda x: np.sqrt(mean_squared_error(x["total_enrol"], x["pred"])))
        .sort_values(ascending=False)
    )

    print("\nTop 10 states by RMSE:")
    print(error_by_state.head(10))

    return model, preds, error_by_state


def monthly_backtest(df, min_train_months=3):
    df = df.copy()
    df["year_month"] = df["date"].dt.to_period("M")

    months = sorted(df["year_month"].unique())

    all_metrics = []
    state_errors_all = []

    for i in range(min_train_months, len(months)):
        train_months = months[:i]
        test_month = months[i]

        train_df = df[df["year_month"].isin(train_months)]
        test_df = df[df["year_month"] == test_month]

        if test_df.empty or train_df.empty:
            continue

        print(
            f"\n=== Train: {train_months[0]} → {train_months[-1]} | Test: {test_month} ==="
        )

        model, preds, state_err = train_and_evaluate(
            train_df, test_df, label=f"Monthly Backtest: {test_month}"
        )

        # Collect metrics
        rmse = np.sqrt(mean_squared_error(test_df["total_enrol"], preds))
        mae = mean_absolute_error(test_df["total_enrol"], preds)
        mape = safe_mape(test_df["total_enrol"], preds)

        all_metrics.append(
            {"test_month": str(test_month), "rmse": rmse, "mae": mae, "mape": mape}
        )

        state_err.name = str(test_month)
        state_errors_all.append(state_err)
        log_header(f"BACKTEST — {test_month}")
        log_text(f"Train months: {train_months[0]} → {train_months[-1]}")

        log_dict({"rmse": rmse, "mae": mae, "mape": mape})

    metrics_df = pd.DataFrame(all_metrics)

    log_header("BACKTEST — AGGREGATE METRICS")
    log_text(metrics_df.describe().to_string())
    print("\n=== AGGREGATED MONTHLY BACKTEST METRICS ===")
    print(metrics_df.describe())

    def forecast_stability_index(preds):
        return np.std(preds) / (np.mean(preds) + 1e-6)

    print(forecast_stability_index(preds))
    return metrics_df, state_errors_all


def train_final_model(df):
    X = df[FEATURES]
    y = np.log1p(df["total_enrol"])

    pool = Pool(X, y, cat_features=CAT_FEATURES)

    model = CatBoostRegressor(
        loss_function="Poisson",
        iterations=600,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=100,
    )

    model.fit(pool)

    return model


def forecast_2_years(df, final_cb_model, horizon_days=730):
    """
    Recursive per state daily forecast for 2 years
    """

    history = df.copy()
    forecasts = []

    last_date = history["date"].max()
    states = history["state"].unique()

    for step in range(horizon_days):
        current_date = last_date + pd.Timedelta(days=step + 1)

        rows = []

        for state in states:
            state_hist = history[history["state"] == state].iloc[-7:]

            if len(state_hist) < 7:
                continue

            row = {
                "date": current_date,
                "state": state,
                "district": state_hist["district"].mode()[0],
                "pincode": state_hist["pincode"].mode()[0],
                "dow": current_date.dayofweek,
                "is_weekend": int(current_date.dayofweek >= 5),
                "month_sin": np.sin(2 * np.pi * current_date.month / 12),
                "month_cos": np.cos(2 * np.pi * current_date.month / 12),
                "lag_1": state_hist.iloc[-1]["total_enrol"],
                "lag_7": state_hist.iloc[0]["total_enrol"],
                "rolling_7": state_hist["total_enrol"].mean(),
                "child_ratio": state_hist["child_ratio"].mean(),
                "adult_ratio": state_hist["adult_ratio"].mean(),
            }

            rows.append(row)

        future_day_df = pd.DataFrame(rows)

        pool = Pool(future_day_df[FEATURES], cat_features=CAT_FEATURES)

        preds = np.expm1(final_cb_model.predict(pool)).clip(min=0)

        future_day_df["total_enrol"] = preds
        forecasts.append(future_day_df)

        history = pd.concat([history, future_day_df], ignore_index=True)

    future_df = pd.concat(forecasts).reset_index(drop=True)
    return future_df


def plot_aggregate(df, title):
    agg = df.groupby("date")["total_enrol"].sum()
    plt.figure(figsize=(14, 5))
    plt.plot(agg.index, agg.values)
    plt.title(title)
    plt.show()


def plot_feature_importance(model):
    fi = model.get_feature_importance(prettified=True)
    print("\nFeature Importance:")
    print(fi)

    plt.figure(figsize=(10, 6))
    plt.barh(fi["Feature Id"], fi["Importances"])
    plt.title("CatBoost Feature Importance")
    plt.gca().invert_yaxis()
    plt.show()


def main_sf():
    df = load_and_preprocess("./merged_second_iter.csv")
    plot_aggregate(df, "Historical Total Enrolments (All States)")

    metrics_df, state_errors = monthly_backtest(df)

    metrics_df.to_csv("monthly_backtest_metrics.csv", index=False)
    final_cb_model = train_final_model(df)

    assert hasattr(final_cb_model, "get_feature_importance")

    fi = final_cb_model.get_feature_importance(prettified=True)

    log_header("FINAL MODEL TRAINED")
    log_text(f"Training rows: {len(df)}")
    log_header("FEATURE IMPORTANCE")
    log_text(fi.to_string(index=False))

    print(fi)


def main():
    reset_logs()

    df = load_and_preprocess("merged.csv")

    log_header("DATASET SUMMARY")
    log_text(f"Rows: {len(df)}")
    log_text(f"Date range: {df['date'].min()} → {df['date'].max()}")
    log_text(f"States: {df['state'].nunique()}")

    metrics_df, state_errors = monthly_backtest(df)

    log_header("MONTHLY BACKTEST METRICS")
    log_text(metrics_df.to_string(index=False))

    final_cb_model = train_final_model(df)

    assert hasattr(final_cb_model, "get_feature_importance")

    fi = final_cb_model.get_feature_importance(prettified=True)
    log_header("FEATURE IMPORTANCE")
    log_text(fi.to_string(index=False))

    # 3️⃣ Forecast
    future_df = forecast_2_years(df, final_cb_model)

    future_df.to_csv("forecast_2yr.csv", index=False)

    log_header("2-YEAR FORECAST SUMMARY")
    log_text(f"Forecast rows: {len(future_df)}")
    log_text(f"Forecast range: {future_df['date'].min()} → {future_df['date'].max()}")
    log_text(f"Total predicted enrolments: {future_df['total_enrol'].sum():,.0f}")

    monthly_forecast = future_df.groupby(pd.Grouper(key="date", freq="M"))[
        "total_enrol"
    ].sum()

    log_text("\nMONTHLY FORECAST TOTALS:")
    log_text(monthly_forecast.to_string())


if __name__ == "__main__":
    main()
