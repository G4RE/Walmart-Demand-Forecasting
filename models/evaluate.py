"""
Evaluate the trained XGBoost model: WMAE, a scale-independent percentage
error, feature importance, and actual vs predicted plots for a few
representative store/dept combinations.

Run: python models/evaluate.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

from etl.load_data import load_data
from models.features import FEATURE_COLUMNS
from models.train import prepare, split, wmae

MODEL_PATH = "models/artifacts/xgb_model.json"


def load_model() -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model


def wape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error: total error as a % of total sales.
    Unlike WMAE, this is scale-independent, so it can be compared across
    departments of very different sales volumes.
    """
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100)


def print_error_summary(test: pd.DataFrame, preds: np.ndarray) -> None:
    overall_wmae = wmae(test["Weekly_Sales"], preds, test["IsHoliday"])
    overall_wape = wape(test["Weekly_Sales"], preds)
    print(f"Overall WMAE: {overall_wmae:,.2f}")
    print(f"Overall WAPE: {overall_wape:.1f}% of total sales\n")

    # Break out by sales volume tercile, since a single average WMAE can
    # hide big departments doing well masking small departments doing badly.
    test = test.copy()
    test["_pred"] = preds
    tercile_edges = test["Weekly_Sales"].quantile([0, 1 / 3, 2 / 3, 1]).values
    labels = ["Low volume", "Mid volume", "High volume"]
    test["_tier"] = pd.cut(test["Weekly_Sales"], bins=tercile_edges,
                            labels=labels, include_lowest=True)

    print("WAPE by sales volume tier:")
    for tier in labels:
        subset = test[test["_tier"] == tier]
        tier_wape = wape(subset["Weekly_Sales"], subset["_pred"])
        avg_sales = subset["Weekly_Sales"].mean()
        print(f"  {tier:<12} (avg ${avg_sales:,.0f}/week): {tier_wape:.1f}% WAPE")
    print()


def plot_feature_importance(model: xgb.XGBRegressor) -> None:
    importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1]))

    plt.figure(figsize=(8, 6))
    plt.barh(list(importance.keys()), list(importance.values()))
    plt.title("XGBoost Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("models/artifacts/feature_importance.png")
    print("Saved models/artifacts/feature_importance.png")


def plot_actual_vs_predicted(model, test, store: int, dept: int) -> None:
    subset = test[(test["Store"] == store) & (test["Dept"] == dept)].sort_values("Date")
    if subset.empty:
        print(f"No test data for Store {store}, Dept {dept} - skipping")
        return

    preds = model.predict(subset[FEATURE_COLUMNS])

    plt.figure(figsize=(10, 4))
    plt.plot(subset["Date"], subset["Weekly_Sales"], label="Actual", marker="o")
    plt.plot(subset["Date"], preds, label="Predicted", marker="x")
    plt.title(f"Store {store}, Dept {dept} - Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Weekly Sales")
    plt.legend()
    plt.tight_layout()
    filename = f"models/artifacts/actual_vs_predicted_store{store}_dept{dept}.png"
    plt.savefig(filename)
    print(f"Saved {filename}")


def main() -> None:
    model = load_model()

    df = load_data()
    df = prepare(df)
    _, test = split(df)

    preds = model.predict(test[FEATURE_COLUMNS])
    print_error_summary(test, preds)

    plot_feature_importance(model)

    # A couple of representative store/dept combos - adjust to taste once
    # you've looked at which ones have interesting patterns.
    for store, dept in [(1, 1), (20, 7)]:
        plot_actual_vs_predicted(model, test, store, dept)


if __name__ == "__main__":
    main()
