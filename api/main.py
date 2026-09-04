"""
FastAPI app serving forecasts from the trained model.

Serves predictions for (store, dept, date) combinations that exist in the
dataset - this is a portfolio demo of wrapping a trained model behind an
API, not a live forecasting service for arbitrary future dates (that would
need a separate multi-step-ahead forecasting approach).

Run: uvicorn api.main:app --reload
Then: http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager
from datetime import date

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException

from etl.load_data import load_data
from models.features import FEATURE_COLUMNS
from models.train import prepare

MODEL_PATH = "models/artifacts/xgb_model.json"

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    df = load_data()
    state["model"] = model
    state["data"] = prepare(df)

    yield
    state.clear()


app = FastAPI(
    title="Walmart Sales Forecast API",
    description="Serves weekly sales forecasts from a trained XGBoost model.",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": "model" in state}


@app.get("/stores")
def list_stores():
    """All store IDs present in the dataset."""
    stores = sorted(state["data"]["Store"].unique().tolist())
    return {"count": len(stores), "stores": stores}


@app.get("/stores/{store}/depts")
def list_depts(store: int):
    """Department IDs present for a given store."""
    subset = state["data"][state["data"]["Store"] == store]
    if subset.empty:
        raise HTTPException(status_code=404, detail=f"No data for Store {store}.")

    depts = sorted(subset["Dept"].unique().tolist())
    return {"store": store, "count": len(depts), "depts": depts}


@app.get("/stores/{store}/depts/{dept}/dates")
def list_dates(store: int, dept: int):
    """Dates available for a given store/dept, so callers know what /forecast will accept."""
    subset = state["data"][
        (state["data"]["Store"] == store) & (state["data"]["Dept"] == dept)
    ]
    if subset.empty:
        raise HTTPException(
            status_code=404, detail=f"No data for Store {store}, Dept {dept}."
        )

    dates = sorted(subset["Date"].dt.date.astype(str).tolist())
    return {"store": store, "dept": dept, "count": len(dates), "dates": dates}


@app.get("/forecast")
def forecast(store: int, dept: int, forecast_date: date):
    data = state["data"]
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
                "dataset (2010-2012), not arbitrary future dates. "
                f"See /stores/{store}/depts/{dept}/dates for valid dates."
            ),
        )

    prediction = state["model"].predict(row[FEATURE_COLUMNS])[0]
    actual = row["Weekly_Sales"].values[0]

    return {
        "store": store,
        "dept": dept,
        "date": str(forecast_date),
        "predicted_sales": round(float(prediction), 2),
        "actual_sales": round(float(actual), 2),
    }
