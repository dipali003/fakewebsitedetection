"""End-to-end pipeline: data → splits → train 3 models → evaluate → artifacts."""
from __future__ import annotations

import json

import numpy as np

import config
from src import evaluate as ev
from src.data_loader import load_email_dataset, load_url_dataset, make_pair_dataset
from src.preprocess import stratified_splits
from src.train import (make_pair_splits, train_email_model, train_fusion_model,
                       train_url_model)


def run_pipeline(quick: bool = False) -> dict:
    """Train and evaluate all three models; persist metrics and plots.

    ``quick=True`` shrinks the synthetic corpora for a fast smoke test.
    """
    if quick:
        config.SYNTH_URL_ROWS = min(config.SYNTH_URL_ROWS, 1600)
        config.SYNTH_EMAIL_ROWS = min(config.SYNTH_EMAIL_ROWS, 900)
        config.MAX_EPOCHS = 3

    # ---------------------------------------------------------------- data
    url_df = load_url_dataset()
    email_df = load_email_dataset()
    url_df.to_csv(config.PROCESSED_DIR / "urls.csv", index=False)
    email_df.to_csv(config.PROCESSED_DIR / "emails.csv", index=False)
    pair_df = make_pair_dataset(url_df, email_df)

    splits = stratified_splits(url_df, email_df)
    pair_splits = make_pair_splits(pair_df)

    # ---------------------------------------------------------------- train
    url_model, _ = train_url_model(splits)
    email_model, _, email_pre = train_email_model(splits)
    fusion_model, _ = train_fusion_model(url_model, email_model, pair_df,
                                         email_pre, pair_splits)

    # ---------------------------------------------------------------- evaluate
    results: dict = {}

    from src.preprocess import encode_url_chars
    u_x = encode_url_chars(list(splits["url_test"][0]))
    u_proba = url_model.predict(u_x, verbose=0).ravel()
    results["url_cnn"] = ev.evaluate_binary(splits["url_test"][1], u_proba)
    ev.plot_confusion(splits["url_test"][1], u_proba, "URL CNN — test", "url_confusion.png")
    ev.plot_roc(splits["url_test"][1], u_proba, "URL CNN — test", "url_roc.png")

    e_x = email_pre.transform(list(splits["email_test"][0]))
    e_proba = email_model.predict(e_x, verbose=0).ravel()
    results["email_bilstm"] = ev.evaluate_binary(splits["email_test"][1], e_proba)
    ev.plot_confusion(splits["email_test"][1], e_proba, "Email BiLSTM — test", "email_confusion.png")
    ev.plot_roc(splits["email_test"][1], e_proba, "Email BiLSTM — test", "email_roc.png")

    p_rows = pair_df.iloc[pair_splits["test"]]
    f_proba = fusion_model.predict(
        [encode_url_chars(list(p_rows["url"])),
         email_pre.transform(list(p_rows["text"]))], verbose=0).ravel()
    results["fusion"] = ev.evaluate_binary(p_rows["label"], f_proba)
    ev.plot_confusion(p_rows["label"], f_proba, "Fusion model — test", "fusion_confusion.png")
    ev.plot_roc(p_rows["label"], f_proba, "Fusion model — test", "fusion_roc.png")

    for name, hist_file in [("url_cnn", "history_url.json"),
                            ("email_bilstm", "history_email.json"),
                            ("fusion", "history_fusion.json")]:
        hist_path = config.ARTIFACTS_DIR / hist_file
        if hist_path.exists():
            ev.plot_training_history(json.loads(hist_path.read_text()),
                                     name, f"{name}_history.png")

    out = config.ARTIFACTS_DIR / "metrics.json"
    out.write_text(json.dumps(results, indent=2))
    print("\n================ TEST RESULTS ================")
    for name, m in results.items():
        print(f"{name:14s} acc={m['accuracy']:.3f}  prec={m['precision']:.3f}  "
              f"rec={m['recall']:.3f}  f1={m['f1']:.3f}  "
              f"auc={m['roc_auc'] if m['roc_auc'] is not None else float('nan'):.3f}")
    print(f"Metrics written -> {out.relative_to(config.ROOT)}")
    return results
