import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_CSV_PATH = os.path.join(DATA_DIR, "Pakistan_Weather_Cleaned.csv")

OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUTS_DIR, "checkpoints")
EDA_DIR = os.path.join(OUTPUTS_DIR, "eda")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")

for d in [DATA_DIR, CHECKPOINT_DIR, EDA_DIR, PLOTS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Processed data files (all live under data/processed/)
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

VOCAB_PATH = os.path.join(PROCESSED_DIR, "vocab.json")
THRESHOLDS_PATH = os.path.join(PROCESSED_DIR, "category_thresholds.json")

TRAIN_LM_CSV = os.path.join(PROCESSED_DIR, "train.csv")
VAL_LM_CSV = os.path.join(PROCESSED_DIR, "val.csv")
TEST_LM_CSV = os.path.join(PROCESSED_DIR, "test.csv")

TRAIN_FORECAST_CSV = os.path.join(PROCESSED_DIR, "forecast_train.csv")
VAL_FORECAST_CSV = os.path.join(PROCESSED_DIR, "forecast_val.csv")
TEST_FORECAST_CSV = os.path.join(PROCESSED_DIR, "forecast_test.csv")

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
LOSS_HISTORY_PATH = os.path.join(CHECKPOINT_DIR, "loss_history.json")

DATE_COLUMN = "Observation_Date"
NUMERIC_FEATURE_COLUMNS = [
    "Average_Temperature_C",
    "Maximum_Temperature_C",
    "Minimum_Temperature_C",
    "Relative_Humidity",
]

WINDOW_DAYS = 5  
TEMP_LABELS = {"Cold": 0, "Moderate": 1, "Hot": 2}
HUMIDITY_LABELS = {"Dry": 0, "Moderate": 1, "Humid": 2}
INV_TEMP_LABELS = {v: k for k, v in TEMP_LABELS.items()}
INV_HUMIDITY_LABELS = {v: k for k, v in HUMIDITY_LABELS.items()}

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1  

MAX_SEQ_LEN = 64      
CLS_SEQ_LEN = 256     
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
D_FF = 512
DROPOUT = 0.15       

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 2e-4  
WEIGHT_DECAY = 0.02   
LM_LOSS_WEIGHT = 1.0
CLS_LOSS_WEIGHT = 0.5
GRAD_CLIP_NORM = 1.0
