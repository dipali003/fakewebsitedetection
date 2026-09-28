"""Central configuration for the deep-learning phishing detection system."""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # user-supplied CSVs (optional)
PROCESSED_DIR = DATA_DIR / "processed"  # generated/normalised datasets
ARTIFACTS_DIR = ROOT / "artifacts"    # models, tokenizers, metrics, plots

for _d in (RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- reproducibility
SEED = 42

# ---------------------------------------------------------------- URL model (char-level CNN)
URL_MAX_LEN = 200        # characters per URL
URL_EMB_DIM = 48
URL_FILTERS = 128
URL_KERNEL = 5

# ---------------------------------------------------------------- Email model (word-level BiLSTM)
EMAIL_MAX_TOKENS = 220
EMAIL_VOCAB_SIZE = 20000
EMAIL_EMB_DIM = 64
EMAIL_LSTM_UNITS = 64

# ---------------------------------------------------------------- Fusion head
FUSION_DENSE_UNITS = 64

# ---------------------------------------------------------------- Training
BATCH_SIZE = 64
MAX_EPOCHS = 30
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 4
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ---------------------------------------------------------------- Synthetic fallback sizes
SYNTH_URL_ROWS = 6000
SYNTH_EMAIL_ROWS = 3000

# ---------------------------------------------------------------- Labels / inference
LABEL_PHISHING = 1
LABEL_LEGITIMATE = 0
THRESHOLD = 0.5          # sigmoid cut-off
HIGH_RISK = 0.75
MED_RISK = 0.45
