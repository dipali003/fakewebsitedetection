"""Load real phishing datasets when present, otherwise generate synthetic data.

Supported real datasets (place the CSV in ``data/raw/``):

* **URL datasets** — a column named ``url`` plus a label column named one of
  ``label`` / ``class`` / ``result`` / ``type``.  Labels may be strings
  (``phishing``/``benign``/``legitimate``/``spam``/``defacement``/``malware`` …)
  or integers where ``1``/``-1`` denotes phishing and ``0`` legitimate.
* **Email datasets** — a text column (``text`` / ``body`` / ``email`` /
  ``message`` / ``content`` / ``v2``) plus a label column.  ``spam`` maps to
  the phishing class, ``ham`` to legitimate (the SMS-spam convention).

If no CSV is found, deterministic synthetic corpora are generated so the full
train/evaluate/serve pipeline works out of the box.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.synth_data import synthetic_url_frame, synthetic_email_frame

# --------------------------------------------------------------------------
# label normalisation
# --------------------------------------------------------------------------
PHISH_WORDS = {
    "phishing", "phish", "spam", "malicious", "bad", "fraud", "fraudulent",
    "defacement", "malware", "scam", "1", "-1", "yes", "true",
}
LEGIT_WORDS = {
    "legitimate", "legit", "benign", "ham", "good", "safe", "normal",
    "0", "no", "false",
}

TEXT_COLS = ("text", "body", "email", "message", "content", "v2")
URL_COLS = ("url", "urls", "website")


def _find_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lowered = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def normalise_label(value) -> int:
    """Map heterogeneous label formats to 1 (phishing) / 0 (legitimate)."""
    s = str(value).strip().lower()
    if s in PHISH_WORDS:
        return 1
    if s in LEGIT_WORDS:
        return 0
    try:
        num = float(s)
        return 1 if num in (1, -1) else 0
    except ValueError:
        return 0


def load_url_dataset() -> pd.DataFrame:
    """Load a URL CSV from ``data/raw`` if present, else synthesize."""
    for path in sorted(config.RAW_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:                                   # noqa: BLE001
            continue
        url_col = _find_col(df, URL_COLS)
        label_col = _find_col(df, ("label", "class", "result", "type", "status"))
        if url_col and label_col:
            out = pd.DataFrame({
                "url": df[url_col].astype(str),
                "label": df[label_col].map(normalise_label),
            })
            out = out.dropna().drop_duplicates(subset="url")
            if len(out) >= 200:
                print(f"[data] URL dataset loaded from {path.name}: {len(out)} rows")
                return out
    print("[data] No URL CSV found in data/raw - generating synthetic URLs")
    return synthetic_url_frame(config.SYNTH_URL_ROWS, seed=config.SEED)


def load_email_dataset() -> pd.DataFrame:
    """Load an email/SMS CSV from ``data/raw`` if present, else synthesize."""
    for path in sorted(config.RAW_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:                                   # noqa: BLE001
            continue
        text_col = _find_col(df, TEXT_COLS)
        label_col = _find_col(df, ("label", "class", "v1", "target", "category"))
        if text_col and label_col:
            out = pd.DataFrame({
                "text": df[text_col].astype(str),
                "label": df[label_col].map(normalise_label),
            })
            out = out.dropna().drop_duplicates(subset="text")
            if len(out) >= 200:
                print(f"[data] Email dataset loaded from {path.name}: {len(out)} rows")
                return out
    print("[data] No email CSV found in data/raw - generating synthetic emails")
    return synthetic_email_frame(config.SYNTH_EMAIL_ROWS, seed=config.SEED)


def make_pair_dataset(url_df: pd.DataFrame, email_df: pd.DataFrame,
                      n: int | None = None, seed: int = config.SEED) -> pd.DataFrame:
    """Pair each URL with a randomly drawn email body of the *same* class.

    The fusion model learns P(phishing | url, email); pairing within class
    keeps the joint label consistent while random pairing breaks spurious
    correlations between the two modalities.
    """
    rng = random.Random(seed)
    phish_urls = url_df[url_df.label == 1]["url"].tolist()
    legit_urls = url_df[url_df.label == 0]["url"].tolist()
    phish_emails = email_df[email_df.label == 1]["text"].tolist()
    legit_emails = email_df[email_df.label == 0]["text"].tolist()

    n_pos = min(n or len(phish_urls), len(phish_urls))
    n_neg = min(n or len(legit_urls), len(legit_urls))
    rows = []
    for _ in range(n_pos):
        rows.append((rng.choice(phish_urls), rng.choice(phish_emails), 1))
    for _ in range(n_neg):
        rows.append((rng.choice(legit_urls), rng.choice(legit_emails), 0))
    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=["url", "text", "label"])
