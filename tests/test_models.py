"""Model-building tests (TensorFlow required; skipped if unavailable)."""
from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.models import build_email_lstm, build_fusion_model, build_url_cnn  # noqa: E402
from src.preprocess import EmailPreprocessor, encode_url_chars             # noqa: E402


def test_url_cnn_forward_pass():
    m = build_url_cnn()
    x = encode_url_chars(["https://paypal.verify-login.xyz/a", "https://github.com"])
    y = m.predict(x, verbose=0)
    assert y.shape == (2, 1)
    assert ((y >= 0) & (y <= 1)).all()


def test_email_lstm_forward_pass():
    m = build_email_lstm(vocab_size=500, max_tokens=20)
    x = np.random.randint(1, 500, size=(3, 20))
    y = m.predict(x, verbose=0)
    assert y.shape == (3, 1)


def test_fusion_model_wires_both_encoders():
    url_m = build_url_cnn(max_len=50)
    email_m = build_email_lstm(vocab_size=300, max_tokens=12)
    fused = build_fusion_model(url_m, email_m)
    xu = encode_url_chars(["http://a.tk/x", "https://ok.com/y"], max_len=50)
    xe = np.random.randint(1, 300, size=(2, 12))
    y = fused.predict([xu, xe], verbose=0)
    assert y.shape == (2, 1)
    assert fused.count_params() > url_m.count_params()


def test_fusion_one_epoch_training_step():
    url_m = build_url_cnn(max_len=40)
    email_m = build_email_lstm(vocab_size=200, max_tokens=8)
    pre = EmailPreprocessor(vocab_size=200, max_tokens=8).fit(["a b c", "d e f"])
    from src.train import train_fusion_model
    import pandas as pd
    pair_df = pd.DataFrame({
        "url": ["http://a.tk/1", "http://b.tk/2", "https://ok.com", "https://x.org"],
        "text": ["verify now", "urgent update", "meeting notes", "hello"],
        "label": [1, 1, 0, 0],
    })
    splits = {"train": [0, 1, 2, 3], "val": [0, 1, 2, 3], "test": [0, 1, 2, 3]}
    model, _ = train_fusion_model(url_m, email_m, pair_df, pre, splits)
    proba = model.predict([encode_url_chars(list(pair_df["url"]), max_len=40),
                           pre.transform(list(pair_df["text"]))], verbose=0).ravel()
    assert ((proba >= 0) & (proba <= 1)).all()
