"""
Evaluate the trained XGBoost model: feature importance and actual vs
predicted plots for a few representative store/dept combinations.

Run: python models/evaluate.py
"""

import matplotlib.pyplot as plt
import xgboost as xgb

from etl.load_data import load_data
from models.features import FEATURE_COLUMNS
from models.train import prepare, split

MODEL_PATH = "models/artifacts/xgb_model.json"


def load_model() -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model


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

    plot_feature_importance(model)

    # A couple of representative store/dept combos - adjust to taste once
    # you've looked at which ones have interesting patterns.
    for store, dept in [(1, 1), (20, 7)]:
        plot_actual_vs_predicted(model, test, store, dept)


if __name__ == "__main__":
    main()
