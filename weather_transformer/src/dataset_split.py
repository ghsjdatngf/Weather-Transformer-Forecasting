import pandas as pd
from src import config


def chronological_split(df: pd.DataFrame, train_ratio: float = config.TRAIN_RATIO,
                         val_ratio: float = config.VAL_RATIO):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df


if __name__ == "__main__":
    from src.label_creation import add_category_labels, build_forecast_labels
    from src.feature_selection import select_features
    from src.preprocessing import build_text_sequences
    from src.data_cleaning import run as clean_run

    clean_df = clean_run()
    selected_df = select_features(clean_df)
    labeled_df, _ = add_category_labels(selected_df)
    text_df = build_text_sequences(labeled_df)
    forecast_df = build_forecast_labels(text_df)

    train_df, val_df, test_df = chronological_split(text_df)
    f_train, f_val, f_test = chronological_split(forecast_df)

    print(f"LM split       -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    print(f"Forecast split -> train: {len(f_train)}, val: {len(f_val)}, test: {len(f_test)}")
