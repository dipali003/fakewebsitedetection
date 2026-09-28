"""Unit tests for preprocessing and data utilities (no TensorFlow needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import normalise_label
from src.preprocess import EmailPreprocessor, clean_email_text, encode_url_chars
from src.synth_data import synthetic_email_frame, synthetic_url_frame


def test_normalise_label_strings():
    assert normalise_label("phishing") == 1
    assert normalise_label("Spam") == 1
    assert normalise_label("benign") == 0
    assert normalise_label("ham") == 0


def test_normalise_label_numeric():
    assert normalise_label(1) == 1
    assert normalise_label(-1) == 1
    assert normalise_label(0) == 0
    assert normalise_label("0") == 0


def test_synthetic_frames_balanced():
    u = synthetic_url_frame(200, seed=1)
    e = synthetic_email_frame(200, seed=2)
    assert set(u.columns) == {"url", "label"}
    assert set(e.columns) == {"text", "label"}
    assert (u.label == 1).sum() == (u.label == 0).sum() == 100
    assert e.text.str.len().min() > 0


def test_encode_url_chars_shapes_and_values():
    out = encode_url_chars(["https://a.com/x", "https://а.com/ю"])
    assert out.shape == (2, 200)
    assert out.dtype == np.int32
    assert (out >= 0).all() and (out < len("abcdefghijklmnopqrstuvwxyz0123456789./:?=@&%#_-+~абвгдежзиклмнопрстуфхцчіыѕדضصطغاب৫৪") + 2).all()
    # pad region must be zero
    assert (out[0, 20:] == 0).all()


def test_clean_email_text_strips_urls_and_html():
    cleaned = clean_email_text("<p>Visit <a href='http://x.tk'>site</a> now</p>")
    assert "http" not in cleaned and "<" not in cleaned


def test_email_preprocessor_roundtrip():
    texts = ["your account is suspended", "meeting notes attached", "hello world"] * 5
    pre = EmailPreprocessor(vocab_size=100, max_tokens=10).fit(texts)
    arr = pre.transform(["your account is suspended now"])
    assert arr.shape == (1, 10)
    assert arr[0][0] != 0  # non-empty sequence starts with a token id
