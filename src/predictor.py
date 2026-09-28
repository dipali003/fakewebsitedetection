"""Inference: load trained artifacts and score URLs / emails / pairs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

import config
from src.preprocess import EmailPreprocessor, encode_url_chars


class ArtifactError(RuntimeError):
    """Raised when required trained artifacts are missing."""


def _load(path: Path) -> tf.keras.Model:
    if not Path(path).exists():
        raise ArtifactError(
            f"Model not found: {path}. Run `python main.py train` first.")
    return tf.keras.models.load_model(path, compile=False)


class Predictor:
    """Convenience wrapper around the three trained models."""

    def __init__(self, artifacts_dir: Path | None = None):
        base = Path(artifacts_dir) if artifacts_dir else config.ARTIFACTS_DIR
        self.url_model = _load(base / "url_cnn.keras")
        self.email_model = _load(base / "email_bilstm.keras")
        self.fusion_model = _load(base / "fusion_model.keras")
        self.email_pre = EmailPreprocessor.load(base / "email_preprocessor.json")

    # ------------------------------------------------------------------
    def score_url(self, url: str) -> float:
        x = encode_url_chars([url])
        return float(self.url_model.predict(x, verbose=0)[0][0])

    def score_email(self, text: str) -> float:
        x = self.email_pre.transform([text])
        return float(self.email_model.predict(x, verbose=0)[0][0])

    def score_pair(self, url: str, text: str) -> tuple[float, float, float]:
        """Return (fusion, url, email) probabilities for one message."""
        xu = encode_url_chars([url])
        xe = self.email_pre.transform([text])
        fusion = float(self.fusion_model.predict([xu, xe], verbose=0)[0][0])
        return fusion, self.score_url(url), self.score_email(text)

    # ------------------------------------------------------------------
    def classify(self, url: str, text: str) -> dict:
        """Full verdict with human-readable risk band."""
        fusion, p_url, p_email = self.score_pair(url, text)
        if fusion >= config.HIGH_RISK:
            risk = "HIGH"
        elif fusion >= config.MED_RISK:
            risk = "MEDIUM"
        else:
            risk = "LOW"
        return {
            "phishing_probability": round(fusion, 4),
            "url_probability": round(p_url, 4),
            "email_probability": round(p_email, 4),
            "is_phishing": fusion >= config.THRESHOLD,
            "risk_level": risk,
        }
