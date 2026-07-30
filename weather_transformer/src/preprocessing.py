import pandas as pd
import torch
from torch.utils.data import Dataset
from src import config

MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]


def record_to_text(row) -> str:
    return (
        f"On {MONTH_NAMES[row['Month'] - 1]} {row['Day']}, {row['Year']}, "
        f"the average temperature was {row['Average_Temperature_C']:.1f} degrees Celsius, "
        f"with a maximum of {row['Maximum_Temperature_C']:.1f} and a minimum of "
        f"{row['Minimum_Temperature_C']:.1f} degrees. "
        f"Relative humidity was {row['Relative_Humidity']:.1f} percent. "
        f"The weather was {row['Temp_Category'].lower()} and {row['Humidity_Category'].lower()}."
    )


def build_text_sequences(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_sequence"] = df.apply(record_to_text, axis=1)
    return df


class WeatherDataset(Dataset):
    """Bundles two DIFFERENT, non-leaking objectives per sample:

    1. Language modeling: today's own text_sequence -> next-token prediction.
    2. Forecasting: the past WINDOW_DAYS context_text -> NEXT day's category
       (loaded from the separate forecast_{split}.csv, aligned by row position).
    """

    def __init__(self, lm_csv_path, forecast_csv_path, tokenizer,
                 max_seq_len=config.MAX_SEQ_LEN, cls_seq_len=config.CLS_SEQ_LEN):
        self.lm_df = pd.read_csv(lm_csv_path).reset_index(drop=True)
        self.forecast_df = pd.read_csv(forecast_csv_path).reset_index(drop=True)
        self.n = min(len(self.lm_df), len(self.forecast_df))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.cls_seq_len = cls_seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        lm_row = self.lm_df.iloc[idx]
        ids = self.tokenizer.encode_fixed(lm_row["text_sequence"], self.max_seq_len)
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)

        f_row = self.forecast_df.iloc[idx]
        context_ids = torch.tensor(
            self.tokenizer.encode_fixed(f_row["context_text"], self.cls_seq_len), dtype=torch.long
        )
        temp_label = torch.tensor(config.TEMP_LABELS[f_row["target_temp_category"]], dtype=torch.long)
        humidity_label = torch.tensor(config.HUMIDITY_LABELS[f_row["target_humidity_category"]], dtype=torch.long)

        return input_ids, target_ids, context_ids, temp_label, humidity_label


if __name__ == "__main__":
    from src.label_creation import add_category_labels
    from src.feature_selection import select_features
    from src.data_cleaning import run as clean_run

    clean_df = clean_run()
    selected_df = select_features(clean_df)
    labeled_df, _ = add_category_labels(selected_df)
    text_df = build_text_sequences(labeled_df)

    print(text_df["text_sequence"].iloc[0])
