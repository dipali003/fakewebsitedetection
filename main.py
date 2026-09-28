#!/usr/bin/env python
"""Command-line entry point for the phishing detection system.

Usage:
    python main.py train            # full pipeline: data → train → evaluate
    python main.py train --quick    # fast smoke run on a small synthetic set
    python main.py predict --url <url> --email "<text>"
    python main.py url --url <url>
    python main.py email --text "<text>"
    python main.py app              # launch the Flask demo web app
"""
from __future__ import annotations

import argparse
import json
import sys

import config


def cmd_train(args) -> int:
    from src.pipeline import run_pipeline
    run_pipeline(quick=args.quick)
    return 0


def _predictor():
    from src.predictor import Predictor
    return Predictor()


def cmd_predict(args) -> int:
    if not (args.url or args.email):
        print("Provide --url and/or --email text.", file=sys.stderr)
        return 2
    p = _predictor()
    result = p.classify(args.url or "", args.email or "")
    print(json.dumps(result, indent=2))
    return 0


def cmd_url(args) -> int:
    p = _predictor()
    prob = p.score_url(args.url)
    print(json.dumps({"url": args.url, "phishing_probability": round(prob, 4),
                      "is_phishing": prob >= config.THRESHOLD}, indent=2))
    return 0


def cmd_email(args) -> int:
    p = _predictor()
    prob = p.score_email(args.text)
    print(json.dumps({"email": args.text[:120], "phishing_probability": round(prob, 4),
                      "is_phishing": prob >= config.THRESHOLD}, indent=2))
    return 0


def cmd_app(args) -> int:
    from app import create_app
    app = create_app()
    print(" * Demo UI -> http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
    return 0


def main() -> int:
    # Windows consoles default to cp1252; force UTF-8 so unicode never crashes prints
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Deep-learning phishing detection system")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train and evaluate all models")
    t.add_argument("--quick", action="store_true", help="small fast smoke run")

    pr = sub.add_parser("predict", help="score a URL + email pair")
    pr.add_argument("--url", default="")
    pr.add_argument("--email", default="")

    u = sub.add_parser("url", help="score a single URL")
    u.add_argument("--url", required=True)

    e = sub.add_parser("email", help="score a single email body")
    e.add_argument("--text", required=True)

    sub.add_parser("app", help="run the Flask demo app")

    args = ap.parse_args()
    return {"train": cmd_train, "predict": cmd_predict, "url": cmd_url,
            "email": cmd_email, "app": cmd_app}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
