"""Training routines for the URL CNN, email BiLSTM and fusion models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import config
from src.models import build_email_lstm, build_fusion_model, build_url_cnn
from src.preprocess import EmailPreprocessor, encode_url_chars


def _callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=config.EARLY_STOP_PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                          min_lr=1e-5, verbose=1),
    ]


def _save_history(history, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    path.write_text(json.dumps(payload, indent=2))


def _as_y(y) -> np.ndarray:
    return np.asarray(y, dtype="int32").reshape(-1, 1)


# --------------------------------------------------------------------------- URL CNN
def train_url_model(splits: dict, save_path: Path | None = None):
    x_tr = encode_url_chars(list(splits["url_train"][0]))
    y_tr = _as_y(splits["url_train"][1])
    x_val = encode_url_chars(list(splits["url_val"][0]))
    y_val = _as_y(splits["url_val"][1])

    model = build_url_cnn()
    hist = model.fit(x_tr, y_tr, validation_data=(x_val, y_val),
                     epochs=config.MAX_EPOCHS, batch_size=config.BATCH_SIZE,
                     callbacks=_callbacks(), verbose=2)
    save_path = Path(save_path or (config.ARTIFACTS_DIR / "url_cnn.keras"))
    model.save(save_path)
    _save_history(hist, config.ARTIFACTS_DIR / "history_url.json")
    print(f"[train] URL CNN saved -> {save_path.name}")
    return model, hist


# --------------------------------------------------------------------------- Email BiLSTM
def train_email_model(splits: dict, save_path: Path | None = None):
    pre = EmailPreprocessor().fit(list(splits["email_train"][0]))
    x_tr = pre.transform(list(splits["email_train"][0]))
    y_tr = _as_y(splits["email_train"][1])
    x_val = pre.transform(list(splits["email_val"][0]))
    y_val = _as_y(splits["email_val"][1])

    model = build_email_lstm()
    hist = model.fit(x_tr, y_tr, validation_data=(x_val, y_val),
                     epochs=config.MAX_EPOCHS, batch_size=config.BATCH_SIZE,
                     callbacks=_callbacks(), verbose=2)
    save_path = Path(save_path or (config.ARTIFACTS_DIR / "email_bilstm.keras"))
    model.save(save_path)
    pre.save(config.ARTIFACTS_DIR / "email_preprocessor.json")
    _save_history(hist, config.ARTIFACTS_DIR / "history_email.json")
    print(f"[train] Email BiLSTM saved -> {save_path.name}")
    return model, hist, pre


# --------------------------------------------------------------------------- Fusion
def make_pair_splits(pair_df, test_size=config.TEST_SPLIT,
                     val_size=config.VAL_SPLIT, seed=config.SEED):
    idx = np.arange(len(pair_df))
    tr, tmp = train_test_split(idx, test_size=test_size + val_size,
                               stratify=pair_df["label"], random_state=seed)
    rel = val_size / (test_size + val_size)
    val, test = train_test_split(tmp, test_size=rel,
                                 stratify=pair_df.iloc[tmp]["label"],
                                 random_state=seed)
    return {"train": tr, "val": val, "test": test}


def train_fusion_model(url_model, email_model, pair_df, email_pre: EmailPreprocessor,
                       pair_splits: dict, save_path: Path | None = None):
    # honour the URL encoder's configured input length
    url_shape = url_model.input_shape
    url_max = int((url_shape[0] if isinstance(url_shape, list) else url_shape)[1])

    def _enc(rows):
        xs = encode_url_chars(list(pair_df.iloc[rows]["url"]), max_len=url_max)
        xe = email_pre.transform(list(pair_df.iloc[rows]["text"]))
        return [xs, xe]

    x_tr, y_tr = _enc(pair_splits["train"]), _as_y(pair_df.iloc[pair_splits["train"]]["label"])
    x_val, y_val = _enc(pair_splits["val"]), _as_y(pair_df.iloc[pair_splits["val"]]["label"])

    model = build_fusion_model(url_model, email_model)
    hist = model.fit(x_tr, y_tr, validation_data=(x_val, y_val),
                     epochs=config.MAX_EPOCHS, batch_size=config.BATCH_SIZE,
                     callbacks=_callbacks(), verbose=2)
    save_path = Path(save_path or (config.ARTIFACTS_DIR / "fusion_model.keras"))
    model.save(save_path)
    _save_history(hist, config.ARTIFACTS_DIR / "history_fusion.json")
    print(f"[train] Fusion model saved -> {save_path.name}")
    return model, hist
