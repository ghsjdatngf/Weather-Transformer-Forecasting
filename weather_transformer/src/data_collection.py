import os
import pandas as pd
from src import config


def load_raw_data(path: str = config.RAW_CSV_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Place Pakistan_Weather_Cleaned.csv in data/."
        )
    df = pd.read_csv(path, parse_dates=[config.DATE_COLUMN])
    df = df.sort_values(config.DATE_COLUMN).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df)} records from {config.RAW_CSV_PATH}")
    print(f"Date range: {df[config.DATE_COLUMN].min().date()} to {df[config.DATE_COLUMN].max().date()}")
    print(df.head())
