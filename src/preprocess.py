"""Preprocessing: char-level URL encoding, word-level email tokenisation, splits.

Transforms raw dataframes into padded numeric arrays ready for the Keras
models, and produces train/val/test splits shared by all three models.

The email tokeniser is implemented from scratch (whitespace words + vocab
dict) so it does not depend on the removed ``keras.preprocessing.text``
legacy API and behaves identically across TF/Keras versions.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config

# char vocabulary used by the URL CNN (0 = pad, 1 = oov, 2+ = chars)
CHARS = ("abcdefghijklmnopqrstuvwxyz0123456789./:?=@&%#_-+~"
         "абвгдежзиклмнопрстуфхціыѕדضصطغ۵৪")
CHAR2IDX = {c: i + 2 for i, c in enumerate(CHARS)}

EMAIL_TAG_RE = re.compile(r"<[^>]+>")
EMAIL_URL_RE = re.compile(r"https?://\S+")
EMAIL_WS_RE = re.compile(r"\s+")


def encode_url_chars(urls, max_len: int = config.URL_MAX_LEN) -> np.ndarray:
    """Map each URL to a fixed-length integer array over ``CHAR2IDX``."""
    out = np.zeros((len(urls), max_len), dtype="int32")
    for i, u in enumerate(urls):
        for j, ch in enumerate(str(u).lower()[:max_len]):
            out[i, j] = CHAR2IDX.get(ch, 1)      # 1 = unknown char
    return out


def clean_email_text(text: str) -> str:
    t = EMAIL_TAG_RE.sub(" ", str(text).lower())
    t = EMAIL_URL_RE.sub(" ", t)                 # remove raw urls from the body
    return EMAIL_WS_RE.sub(" ", t).strip()


class EmailPreprocessor:
    """Fit-once whitespace word tokenizer for the email BiLSTM branch.

    Index conventions: 0 = padding, 1 = OOV, 2+ = most-frequent-first words.
    """

    def __init__(self, vocab_size: int = config.EMAIL_VOCAB_SIZE,
                 max_tokens: int = config.EMAIL_MAX_TOKENS):
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self.word_index: dict[str, int] = {}

    # ------------------------------------------------------------- fit/transform
    def fit(self, texts) -> "EmailPreprocessor":
        counter: Counter = Counter()
        for t in texts:
            counter.update(clean_email_text(t).split())
        # reserve 0=pad, 1=oov → usable ids start at 2
        for idx, (word, _freq) in enumerate(counter.most_common(self.vocab_size - 2)):
            self.word_index[word] = idx + 2
        return self

    def transform(self, texts) -> np.ndarray:
        out = np.zeros((len(texts), self.max_tokens), dtype="int32")
        for i, t in enumerate(texts):
            ids = [self.word_index.get(w, 1) for w in clean_email_text(t).split()]
            ids = ids[: self.max_tokens]
            out[i, : len(ids)] = ids
        return out

    # ------------------------------------------------------------- persistence
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "num_words": self.vocab_size,
            "max_tokens": self.max_tokens,
            "word_index": self.word_index,
        }))

    @classmethod
    def load(cls, path) -> "EmailPreprocessor":
        payload = json.loads(Path(path).read_text())
        obj = cls(vocab_size=payload["num_words"], max_tokens=payload["max_tokens"])
        obj.word_index = payload["word_index"]
        return obj


def stratified_splits(url_df: pd.DataFrame, email_df: pd.DataFrame,
                      test_size: float = config.TEST_SPLIT,
                      val_size: float = config.VAL_SPLIT,
                      seed: int = config.SEED) -> dict[str, Any]:
    """Stratified train/val/test splits for the URL and email frames.

    Returns dict with ``url_*`` / ``email_*`` keys, each holding
    ``(X_series, y_series)``.
    """
    out: dict[str, Any] = {}

    u_tr, u_tmp, utr, utmp = train_test_split(
        url_df["url"], url_df["label"], test_size=test_size + val_size,
        stratify=url_df["label"], random_state=seed)
    rel = val_size / (test_size + val_size)
    u_val, u_test, uval_lab, utest_lab = train_test_split(
        u_tmp, utmp, test_size=rel, stratify=utmp, random_state=seed)
    out["url_train"], out["url_val"], out["url_test"] = (u_tr, utr), (u_val, uval_lab), (u_test, utest_lab)

    e_tr, e_tmp, etr, etmp = train_test_split(
        email_df["text"], email_df["label"], test_size=test_size + val_size,
        stratify=email_df["label"], random_state=seed)
    e_val, e_test, eval_lab, etest_lab = train_test_split(
        e_tmp, etmp, test_size=rel, stratify=etmp, random_state=seed)
    out["email_train"], out["email_val"], out["email_test"] = (e_tr, etr), (e_val, eval_lab), (e_test, etest_lab)
    return out
