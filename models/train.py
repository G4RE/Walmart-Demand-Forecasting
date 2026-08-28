"""
Train and compare a naive baseline against XGBoost for weekly sales forecasting.

Run: python models/train.py
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from etl.load_data import load_data
from models.features import add_features, FEATURE_COLUMNS, TARGET_COLUMN

SPLIT_DATE = "2012-06-01"  # last ~6 months held out as test


def wmae(y_true: pd.Series, y_pred: np.ndarray, is_holiday: pd.Series) -> float:
    """Weighted MAE - the competition's own metric. Holiday weeks weighted 5x."""
    weights = np.where(is_holiday, 5, 1)
    return float(np.average(np.abs(y_true - y_pred), weights=weights))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = add_features(df)
    # Rows without a full year of history can't have sales_lag_52 - drop them
    # rather than imputing, since a guessed lag value would be misleading.
    df = df.dropna(subset=["sales_lag_52", "sales_lag_1", "sales_rolling_mean_4"])
    df["Type"] = df["Type"].astype("category")
    return df


def split(df: pd.DataFrame):
    train = df[df["Date"] < SPLIT_DATE]
    test = df[df["Date"] >= SPLIT_DATE]
    return train, test


def naive_baseline(test: pd.DataFrame) -> np.ndarray:
    """Predict this week's sales as the same week last year."""
    return test["sales_lag_52"].values


def train_xgboost(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        random_state=42,
    )
    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    return model, model.predict(test[FEATURE_COLUMNS])


def main() -> None:
    df = load_data()
    df = prepare(df)
    train, test = split(df)
    print(f"Train: {len(train):,} rows | Test: {len(test):,} rows "
          f"(split at {SPLIT_DATE})")

    baseline_preds = naive_baseline(test)
    baseline_wmae = wmae(test[TARGET_COLUMN], baseline_preds, test["IsHoliday"])
    print(f"Naive baseline WMAE: {baseline_wmae:,.2f}")

    model, xgb_preds = train_xgboost(train, test)
    xgb_wmae = wmae(test[TARGET_COLUMN], xgb_preds, test["IsHoliday"])
    print(f"XGBoost WMAE: {xgb_wmae:,.2f}")

    improvement = (1 - xgb_wmae / baseline_wmae) * 100
    print(f"Improvement over baseline: {improvement:.1f}%")

    model.save_model("models/artifacts/xgb_model.json")
    print("Saved model to models/artifacts/xgb_model.json")


if __name__ == "__main__":
    main()
