import json
import os
import pandas as pd
from src import config


def add_category_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    temp_low, temp_high = df["Average_Temperature_C"].quantile([0.33, 0.66])

    def temp_category(t):
        if t <= temp_low:
            return "Cold"
        elif t <= temp_high:
            return "Moderate"
        return "Hot"

    def humidity_category(h):
        if h < 50:
            return "Dry"
        elif h < 70:
            return "Moderate"
        return "Humid"

    df = df.copy()
    df["Temp_Category"] = df["Average_Temperature_C"].apply(temp_category)
    df["Humidity_Category"] = df["Relative_Humidity"].apply(humidity_category)

    thresholds = {"temp_low": float(temp_low), "temp_high": float(temp_high), "window_days": config.WINDOW_DAYS}
    return df, thresholds


def build_forecast_labels(df: pd.DataFrame, window_days: int = config.WINDOW_DAYS) -> pd.DataFrame:
    """df must already contain 'text_sequence', 'Temp_Category', 'Humidity_Category'."""
    context_texts, target_temp, target_humidity, target_dates = [], [], [], []

    for i in range(window_days, len(df)):
        past_texts = df["text_sequence"].iloc[i - window_days:i].tolist()
        context_texts.append(" ".join(past_texts))
        target_temp.append(df["Temp_Category"].iloc[i])
        target_humidity.append(df["Humidity_Category"].iloc[i])
        target_dates.append(df[config.DATE_COLUMN].iloc[i])

    return pd.DataFrame({
        config.DATE_COLUMN: target_dates,
        "context_text": context_texts,
        "target_temp_category": target_temp,
        "target_humidity_category": target_humidity,
    })


if __name__ == "__main__":
    from src.preprocessing import build_text_sequences
    from src.feature_selection import select_features
    from src.data_cleaning import run as clean_run

    clean_df = clean_run()
    selected_df = select_features(clean_df)
    labeled_df, thresholds = add_category_labels(selected_df)
    labeled_df = build_text_sequences(labeled_df)

    forecast_df = build_forecast_labels(labeled_df)

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    with open(config.THRESHOLDS_PATH, "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"Temp category counts:\n{labeled_df['Temp_Category'].value_counts()}")
    print(f"\nHumidity category counts:\n{labeled_df['Humidity_Category'].value_counts()}")
    print(f"\nForecast samples created: {len(forecast_df)} (window = {config.WINDOW_DAYS} days)")
