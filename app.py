"""Flask demo app for the phishing detection system."""
from __future__ import annotations

import json

from flask import Flask, jsonify, render_template, request

import config


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        payload = request.get_json(silent=True) or {}
        url = str(payload.get("url", "")).strip()
        text = str(payload.get("email", "")).strip()
        if not url and not text:
            return jsonify({"error": "Provide 'url' and/or 'email'."}), 400
        from src.predictor import Predictor
        try:
            p = Predictor()
        except Exception as exc:                      # noqa: BLE001
            return jsonify({"error": str(exc),
                            "hint": "Run `python main.py train` first."}), 503
        result = p.classify(url, text)
        result["input"] = {"url": url, "email_preview": text[:200]}
        return jsonify(result)

    @app.route("/api/health")
    def health():
        required = ["url_cnn.keras", "email_bilstm.keras", "fusion_model.keras",
                    "email_preprocessor.json"]
        ready = all((config.ARTIFACTS_DIR / f).exists() for f in required)
        return jsonify({"status": "ok" if ready else "untrained",
                        "artifacts_dir": str(config.ARTIFACTS_DIR)})

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
