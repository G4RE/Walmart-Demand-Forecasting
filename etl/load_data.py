"""
Load and join the raw Walmart competition CSVs.

Expects stores.csv, train.csv, features.csv in data/
(download from the Kaggle competition page, see README).
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MARKDOWN_COLS = [f"MarkDown{i}" for i in range(1, 6)]


def load_data() -> pd.DataFrame:
    stores = pd.read_csv(DATA_DIR / "stores.csv")
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["Date"])
    features = pd.read_csv(DATA_DIR / "features.csv", parse_dates=["Date"])

    # features.csv duplicates train's IsHoliday column - drop before merging.
    features = features.drop(columns=["IsHoliday"])

    df = train.merge(stores, on="Store", how="left")
    df = df.merge(features, on=["Store", "Date"], how="left")

    # MarkDown columns are mostly null before Nov 2011 (no markdown program yet).
    # Treat missing as "no markdown active" rather than dropping rows.
    df[MARKDOWN_COLS] = df[MARKDOWN_COLS].fillna(0)

    df = df.drop_duplicates(subset=["Store", "Dept", "Date"])
    df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

    return df
