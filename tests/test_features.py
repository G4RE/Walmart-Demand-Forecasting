"""
Regression tests for models.features.add_features.

Focus: lag/rolling calculations must be grouped correctly by (Store, Dept)
so one store/department's history never leaks into another's.
"""

import pandas as pd

from models.features import add_features


def _make_df(rows):
    df = pd.DataFrame(rows, columns=["Store", "Dept", "Date", "Weekly_Sales"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def test_lag_1_shifts_within_group():
    df = _make_df([
        (1, 1, "2011-01-07", 100),
        (1, 1, "2011-01-14", 200),
        (1, 1, "2011-01-21", 300),
    ])
    result = add_features(df)

    assert pd.isna(result.iloc[0]["sales_lag_1"])
    assert result.iloc[1]["sales_lag_1"] == 100
    assert result.iloc[2]["sales_lag_1"] == 200


def test_lag_does_not_leak_across_store_dept_groups():
    df = _make_df([
        (1, 1, "2011-01-07", 100),
        (1, 1, "2011-01-14", 200),
        (2, 1, "2011-01-07", 999),  # different store, same date
        (2, 1, "2011-01-14", 888),
    ])
    result = add_features(df)

    store2 = result[(result["Store"] == 2) & (result["Dept"] == 1)].sort_values("Date")
    # Store 2's first row must not pick up Store 1's sales as its lag.
    assert pd.isna(store2.iloc[0]["sales_lag_1"])
    assert store2.iloc[1]["sales_lag_1"] == 999


def test_lag_52_requires_a_full_year_of_history():
    rows = [(1, 1, f"2011-{(i // 4) + 1:02d}-{(i % 4) * 7 + 1:02d}", 100 + i) for i in range(10)]
    df = _make_df(rows)
    result = add_features(df)

    # With only 10 weeks of history, nothing has a 52-week lag yet.
    assert result["sales_lag_52"].isna().all()


def test_rolling_mean_excludes_current_row():
    df = _make_df([
        (1, 1, "2011-01-07", 100),
        (1, 1, "2011-01-14", 200),
        (1, 1, "2011-01-21", 300),
        (1, 1, "2011-01-28", 400),
        (1, 1, "2011-02-04", 500),
    ])
    result = add_features(df)

    # 4-week rolling mean at the 5th row should average rows 1-4 (100,200,300,400),
    # not include row 5's own value (500).
    assert result.iloc[4]["sales_rolling_mean_4"] == 250
