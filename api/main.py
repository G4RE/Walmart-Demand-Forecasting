"""
FastAPI app serving forecasts from the trained model.

Serves predictions for (store, dept, date) combinations that exist in the
dataset - this is a portfolio demo of wrapping a trained model behind an
API, not a live forecasting service for arbitrary future dates (that would
need a separate multi-step-ahead forecasting approach).

Run: uvicorn api.main:app --reload
Then: http://127.0.0.1:8000/docs
"""

from datetime import date

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException

from etl.load_data import load_data
from models.features import FEATURE_COLUMNS
from models.train import prepare

MODEL_PATH = "models/artifacts/xgb_model.json"

app = FastAPI(
    title="Walmart Sales Forecast API",
    description="Serves weekly sales forecasts from a trained XGBoost model.",
)

model: xgb.XGBRegressor | None = None
data = None


@app.on_event("startup")
def load_model_and_data() -> None:
    global model, data
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    df = load_data()
    data = prepare(df)


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}


@app.get("/forecast")
def forecast(store: int, dept: int, forecast_date: date):
    row = data[
        (data["Store"] == store)
        & (data["Dept"] == dept)
        & (data["Date"] == pd.Timestamp(forecast_date))
    ]

    if row.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No data for Store {store}, Dept {dept} on {forecast_date}. "
                "This API serves forecasts for dates present in the training "
                "dataset (2010-2012), not arbitrary future dates."
            ),
        )

    prediction = model.predict(row[FEATURE_COLUMNS])[0]
    actual = row["Weekly_Sales"].values[0]

    return {
        "store": store,
        "dept": dept,
        "date": str(forecast_date),
        "predicted_sales": round(float(prediction), 2),
        "actual_sales": round(float(actual), 2),
    }
