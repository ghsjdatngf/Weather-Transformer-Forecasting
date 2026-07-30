import json
import os
import pandas as pd
from src import config
from src.data_collection import load_raw_data


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {}

    report["total_records"] = len(df)
    report["missing_values"] = df.isnull().sum().to_dict()
    report["duplicate_dates"] = int(df[config.DATE_COLUMN].duplicated().sum())

    full_range = pd.date_range(df[config.DATE_COLUMN].min(), df[config.DATE_COLUMN].max(), freq="D")
    report["missing_days_in_sequence"] = len(full_range) - len(df)

    logic_violations = df[
        (df["Minimum_Temperature_C"] > df["Average_Temperature_C"])
        | (df["Average_Temperature_C"] > df["Maximum_Temperature_C"])
    ]
    report["logical_inconsistencies"] = int(len(logic_violations))
    report["date_range"] = {
        "start": str(df[config.DATE_COLUMN].min().date()),
        "end": str(df[config.DATE_COLUMN].max().date()),
    }

    df = df.drop_duplicates(subset=[config.DATE_COLUMN])
    df = df.dropna(subset=config.NUMERIC_FEATURE_COLUMNS)
    df = df.sort_values(config.DATE_COLUMN).reset_index(drop=True)

    return df, report


def run():
    df = load_raw_data()
    clean_df, report = clean_data(df)

    report_path = os.path.join(config.EDA_DIR, "cleaning_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 50)
    print("DATA CLEANING REPORT")
    print("=" * 50)
    print(json.dumps(report, indent=2))
    print(f"\nReport saved to {report_path}")

    return clean_df


if __name__ == "__main__":
    run()
