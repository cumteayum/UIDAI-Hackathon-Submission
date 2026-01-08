import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REPORT_PATH = os.path.join(OUTPUT_DIR, "report.txt")


# ===============================
# Logging helper
# ===============================


def log(text):
    with open(REPORT_PATH, "a") as f:
        f.write(text + "\n")


# ===============================
# Load & clean (same hygiene rules)
# ===============================


def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df[df["age_0_5"] != "age_0_5"].reset_index(drop=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["date"])

    for col in ["age_0_5", "age_5_17", "age_18_greater"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ""), errors="coerce"
        )

    df[["age_0_5", "age_5_17", "age_18_greater"]] = df[
        ["age_0_5", "age_5_17", "age_18_greater"]
    ].fillna(0)

    df["total_enrol"] = df["age_0_5"] + df["age_5_17"] + df["age_18_greater"]

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dow"] = df["date"].dt.dayofweek

    return df.sort_values(["state", "date"]).reset_index(drop=True)


# ===============================
# Global summary
# ===============================


def global_summary(df):
    log("=== GLOBAL SUMMARY ===")
    log(f"Date range: {df['date'].min()} → {df['date'].max()}")
    log(f"Total enrolments: {df['total_enrol'].sum():,.0f}")
    log(f"Daily mean enrolment: {df['total_enrol'].mean():.2f}")
    log(f"Daily median enrolment: {df['total_enrol'].median():.2f}")
    log(f"Daily std deviation: {df['total_enrol'].std():.2f}")
    log("")


# ===============================
# Nationwise time series
# ===============================


def plot_nationwise(df):
    national = df.groupby("date")["total_enrol"].sum()

    plt.figure(figsize=(14, 5))
    plt.plot(national.index, national.values)
    plt.title("National Daily Enrolments")
    plt.savefig(f"{OUTPUT_DIR}/national_timeseries.png")
    plt.close()

    monthly = national.resample("M").sum()
    plt.figure(figsize=(14, 5))
    plt.plot(monthly.index, monthly.values)
    plt.title("National Monthly Enrolments")
    plt.savefig(f"{OUTPUT_DIR}/national_monthly.png")
    plt.close()

    log("National peak day enrolment: " + str(national.max()))
    log("")


# ===============================
# Statewise analysis
# ===============================


def statewise_analysis(df):
    state_totals = df.groupby("state")["total_enrol"].sum().sort_values(ascending=False)
    log("=== STATEWISE TOTAL ENROLMENTS ===")
    log(state_totals.to_string())
    log("")

    top_states = state_totals.head(10)

    plt.figure(figsize=(12, 6))
    top_states.plot(kind="bar")
    plt.title("Top 10 States by Total Enrolment")
    plt.savefig(f"{OUTPUT_DIR}/top_states.png")
    plt.close()

    # State volatility
    state_vol = df.groupby("state")["total_enrol"].std().sort_values(ascending=False)

    log("=== STATEWISE VOLATILITY (STD DEV) ===")
    log(state_vol.to_string())
    log("")


def distwise_analysis(df):
    dist_totals = (
        df.groupby("district")["total_enrol"].sum().sort_values(ascending=False)
    )
    log("=== Districtwise TOTAL ENROLMENTS ===")
    log(dist_totals.to_string())
    log("")

    top_states = dist_totals.head(10)

    plt.figure(figsize=(12, 6))
    top_states.plot(kind="bar")
    plt.title("Top 10 States by Total Enrolment")
    plt.savefig(f"{OUTPUT_DIR}/top_dists.png")
    plt.close()

    # State volatility
    dist_vol = df.groupby("district")["total_enrol"].std().sort_values(ascending=False)

    log("=== STATEWISE VOLATILITY (STD DEV) ===")
    log(dist_totals.to_string())
    log("")


# ===============================
# Monthwise seasonality
# ===============================


def monthwise_analysis(df):
    month_avg = df.groupby("month")["total_enrol"].mean()

    plt.figure(figsize=(10, 5))
    month_avg.plot(marker="o")
    plt.title("Average Daily Enrolment by Month")
    plt.savefig(f"{OUTPUT_DIR}/monthwise_avg.png")
    plt.close()

    log("=== MONTHWISE AVERAGE DAILY ENROLMENT ===")
    log(month_avg.to_string())
    log("")


# ===============================
# Age-wise trends
# ===============================


def agewise_analysis(df):
    age_ts = df.groupby("date")[["age_0_5", "age_5_17", "age_18_greater"]].sum()

    plt.figure(figsize=(14, 6))
    plt.plot(age_ts.index, age_ts["age_0_5"], label="Age 0–5")
    plt.plot(age_ts.index, age_ts["age_5_17"], label="Age 5–17")
    plt.plot(age_ts.index, age_ts["age_18_greater"], label="Age 18+")
    plt.legend()
    plt.title("Age-wise Enrolments Over Time")
    plt.savefig(f"{OUTPUT_DIR}/agewise_timeseries.png")
    plt.close()

    log("=== AGE GROUP TOTALS ===")
    log(age_ts.sum().to_string())
    log("")


# ===============================
# Growth rates
# ===============================


def growth_analysis(df):
    monthly = df.groupby(pd.Grouper(key="date", freq="M"))["total_enrol"].sum()
    growth = monthly.pct_change() * 100

    plt.figure(figsize=(14, 5))
    plt.plot(growth.index, growth.values)
    plt.title("Month-on-Month Growth Rate (%)")
    plt.savefig(f"{OUTPUT_DIR}/mom_growth.png")
    plt.close()

    log("=== MONTH-ON-MONTH GROWTH (%) ===")
    log(growth.describe().to_string())
    log("")


# ===============================
# Candlestick-style volatility
# ===============================


def pseudo_candlestick(df):
    daily = df.groupby("date")["total_enrol"].sum()
    ohlc = daily.resample("M").agg(["min", "max", "mean"])

    plt.figure(figsize=(14, 6))
    plt.vlines(ohlc.index, ohlc["min"], ohlc["max"])
    plt.plot(ohlc.index, ohlc["mean"], marker="o")
    plt.title("Monthly Enrolment Range (Min–Max) with Mean")
    plt.savefig(f"{OUTPUT_DIR}/monthly_range.png")
    plt.close()


# ===============================
# Main
# ===============================


def main():
    if os.path.exists(REPORT_PATH):
        os.remove(REPORT_PATH)

    df = load_data("merged.csv")

    global_summary(df)
    plot_nationwise(df)
    statewise_analysis(df)
    distwise_analysis(df)
    monthwise_analysis(df)
    agewise_analysis(df)
    growth_analysis(df)
    pseudo_candlestick(df)

    log("Analysis complete.")
    print("Full analysis generated in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
