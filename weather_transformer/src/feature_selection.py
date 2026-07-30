import json
import os
import pandas as pd
from src import config


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Year"] = df[config.DATE_COLUMN].dt.year
    df["Month"] = df[config.DATE_COLUMN].dt.month
    df["Day"] = df[config.DATE_COLUMN].dt.day

    keep_cols = [config.DATE_COLUMN, "Year", "Month", "Day"] + config.NUMERIC_FEATURE_COLUMNS
    return df[keep_cols]


def report_feature_correlation(df: pd.DataFrame):
    corr = df[config.NUMERIC_FEATURE_COLUMNS].corr()
    path = os.path.join(config.EDA_DIR, "feature_correlation.json")
    with open(path, "w") as f:
        json.dump(corr.round(4).to_dict(), f, indent=2)
    print(f"Feature correlation matrix saved to {path}")
    return corr


if __name__ == "__main__":
    from src.data_cleaning import run as clean_run

    clean_df = clean_run()
    selected_df = select_features(clean_df)
    report_feature_correlation(selected_df)
    print(selected_df.head())
