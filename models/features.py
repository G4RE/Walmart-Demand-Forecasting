"""
Feature engineering for Walmart weekly sales forecasting.

Takes the joined dataset from etl.load_data and adds lag/rolling/calendar
features. All lag and rolling calculations are grouped by (Store, Dept) so
one store/department's history never leaks into another's.
"""

import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Store", "Dept", "Date"]).copy()
    grouped = df.groupby(["Store", "Dept"])["Weekly_Sales"]

    # Same week last year - typically the strongest single predictor here.
    df["sales_lag_52"] = grouped.shift(52)

    # Last week's sales.
    df["sales_lag_1"] = grouped.shift(1)

    # Trailing 4-week average, computed on already-lagged sales so it never
    # includes the current row's own target value.
    df["sales_rolling_mean_4"] = (
        grouped.shift(1).rolling(4).mean().reset_index(level=[0, 1], drop=True)
    )

    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year

    return df


FEATURE_COLUMNS = [
    "Store", "Dept", "IsHoliday", "Size", "Type",
    "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "sales_lag_52", "sales_lag_1", "sales_rolling_mean_4",
    "week_of_year", "month", "year",
]

TARGET_COLUMN = "Weekly_Sales"
