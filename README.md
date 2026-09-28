# 🛡️ Deep Learning Phishing Detection System

Detects phishing URLs **and** phishing email content with a multi-modal
deep-learning pipeline:

| Branch | Input | Architecture | Learns |
|---|---|---|---|
| URL model | raw URL characters | **1-D char-CNN** (2/3/5-gram convolutions) | brand impersonation, homoglyphs, suspicious TLDs, credential keywords, IP hosts |
| Email model | email body / SMS text | **word-level BiLSTM** | urgency, threats, credential-bait phrasing, generic greetings |
| Fusion | both branches | concatenated penultimate features + dense head | joint evidence from URL *and* message |

## Project layout

```
config.py                  hyperparameters & paths
main.py                    CLI: train / predict / url / email / app
app.py                     Flask demo app (JSON API + web UI)
src/
  data_loader.py           real-CSV loading (data/raw) + label normalisation
  synth_data.py            deterministic synthetic corpus (fallback)
  preprocess.py            char encoding, tokenization, stratified splits
  models.py                URL CNN, email BiLSTM, fusion architecture
  train.py                 training loops, callbacks, artifact saving
  evaluate.py              metrics, confusion matrix & ROC plots
  predictor.py             inference wrapper
  pipeline.py              end-to-end orchestration
templates/index.html       demo web UI
tests/                     pytest suite
artifacts/                 trained models, tokenizers, metrics, plots
data/raw/                  ← put real datasets here (optional CSVs)
```

## Quick start

```bash
# 1. Environment
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # Windows
# .venv/bin/pip install -r requirements.txt          # Linux/macOS

# 2. Smoke run (small synthetic data, ~2 min on CPU)
python main.py train --quick

# 3. Full training (synthetic corpus, ~10-20 min on CPU)
python main.py train

# 4. Score a suspicious message
python main.py predict --url "http://paypal-secure.verify-login.tk/account" \
                       --email "Dear customer, your account is suspended, verify now: ..."

# 5. Web demo
python main.py app            # → http://127.0.0.1:5000
```

## Using real datasets

Drop CSVs into `data/raw/` before training — the loader auto-detects them:

* **URLs**: any CSV with a `url` column + label column (`label`/`class`/`result`/`type`).
  String labels (`phishing`, `benign`, `defacement`, `spam`…) or numeric (1/-1 = phish, 0 = legit).
* **Emails**: any CSV with a text column (`text`/`body`/`v2`…) + label column.
  SMS-spam datasets (`ham`/`spam`) work as-is.

Suggested public datasets: Kaggle *Phishing Websites* (UCI-style feature set or raw-URL variants),
Kaggle *SMS Spam Collection*, Enron + Nigerian-fraud corpora, PhishTank feeds,
Nazario phishing corpus.

Artifacts after training (`artifacts/`):
`url_cnn.keras`, `email_bilstm.keras`, `fusion_model.keras`,
`email_preprocessor.json`, `metrics.json`, `history_*.json`, `plots/*.png`.

## Evaluation

`python main.py train` writes `artifacts/metrics.json` and prints:

```
url_cnn        acc=0.99x  prec=0.99x  rec=0.99x  f1=0.99x  auc=0.999
email_bilstm   acc=0.9x   prec=...    rec=...    f1=...    auc=...
fusion         acc=0.9x   prec=...    rec=...    f1=...    auc=...
```

Each model also gets a confusion-matrix and ROC plot in `artifacts/plots/`,
plus loss/accuracy curves — ready to paste into the project report.

## API

```
POST /api/predict   {"url": "...", "email": "..."}   → probability + verdict
GET  /api/health                                     → artifact readiness
```

## Notes

* With no dataset supplied the system trains on a **deterministic synthetic
  corpus**, so the full pipeline is reproducible and CI-friendly. Scores are
  optimistic by design; plug in real data for publication-grade numbers.
* Class probabilities use threshold 0.5; risk bands: ≥0.75 HIGH,
  ≥0.45 MEDIUM, else LOW (see `config.py`).
