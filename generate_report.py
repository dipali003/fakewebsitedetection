"""Generate a comprehensive PDF report for the Phishing Detection System.

Run:  .venv/Scripts/python.exe generate_report.py
Output: artifacts/Phishing_Detection_System_Report.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import config

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = config.ROOT
OUT = config.ARTIFACTS_DIR / "Phishing_Detection_System_Report.pdf"
PLOTS = config.ARTIFACTS_DIR / "plots"

TF_VERSION = "2.21.0"   # matches installed tensorflow-cpu

# ------------------------------------------------------------------ styles
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=22, textColor=colors.HexColor("#1b2631"), spaceAfter=4)
H2 = ParagraphStyle("H2", parent=ss["Heading1"], fontSize=15, textColor=colors.HexColor("#1b4f72"), spaceBefore=14, spaceAfter=6)
H3 = ParagraphStyle("H3", parent=ss["Heading2"], fontSize=12, textColor=colors.HexColor("#1b4f72"), spaceBefore=10, spaceAfter=4)
SUB = ParagraphStyle("Sub", parent=ss["Normal"], fontSize=11, textColor=colors.HexColor("#5d6d7e"), alignment=1, spaceAfter=16)
BODY = ParagraphStyle("Body", parent=ss["Normal"], fontSize=10, leading=14.5, spaceAfter=5)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=2.5)
CODE = ParagraphStyle("Code", parent=ss["Code"], fontSize=8.5, leading=11.5, backColor=colors.HexColor("#f4f6f8"),
                      borderColor=colors.HexColor("#ccd6dd"), borderWidth=0.6, borderPadding=5, spaceAfter=6)
CAP = ParagraphStyle("Cap", parent=ss["Normal"], fontSize=9, textColor=colors.HexColor("#5d6d7e"), alignment=1, spaceBefore=2, spaceAfter=10)

ACCENT = colors.HexColor("#1b4f72")
LIGHT = colors.HexColor("#eaf2f8")
GRID = colors.HexColor("#b8c6d4")


def table(data, widths, header=True, font=8.5):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.5, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                  ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
        for r in range(1, len(data)):
            if r % 2 == 0:
                style.append(("BACKGROUND", (0, r), (-1, r), LIGHT))
    t.setStyle(TableStyle(style))
    return t


def bullets(items):
    return [Paragraph(f"• {x}", BULLET) for x in items]


def img(name, width=150*mm):
    p = PLOTS / name
    if p.exists():
        return Image(str(p), width=width, height=width * 0.42, kind="proportional")
    return Paragraph(f"[plot not found: {name}]", BODY)


story = []

# ================================================================ COVER
story.append(Spacer(1, 60*mm))
story.append(Paragraph("🛡️ Deep Learning<br/>Phishing Detection System", H1))
story.append(Paragraph("Complete Technical Report — Architecture • Pipeline • Tech Stack • Results", SUB))
story.append(Spacer(1, 10*mm))

cover = table([
    ["Project", "Multi-modal Deep Learning Phishing Detection"],
    ["Models", "URL char-CNN + Email BiLSTM + Fusion classifier"],
    ["Framework", f"TensorFlow {TF_VERSION} / Keras 3 (Python 3.13)"],
    ["Demo App", "Flask REST API + Web UI (port 5000)"],
    ["Artifacts", "3 trained .keras models + tokenizer + metrics + plots"],
    ["Status", "Trained & verified end-to-end ✅"],
], [45*mm, 115*mm], header=False, font=9.5)
cover.setStyle(TableStyle([("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold")]))
story.append(cover)

story.append(PageBreak())

# ================================================================ 1. OVERVIEW
story.append(Paragraph("1. System Overview", H2))
story.append(Paragraph(
    "Ye system ek <b>multi-modal deep-learning phishing detector</b> hai jo do alag inputs ko "
    "analyse karta hai — (1) <b>URL</b> ka raw character string, aur (2) <b>Email/SMS body</b> ka text. "
    "Har input ka apna specialised neural network hai, aur dono ke features ko ek "
    "<b>fusion model</b> combine karke final verdict deta hai: phishing probability (0–1), "
    "binary decision (is_phishing), aur risk band (LOW / MEDIUM / HIGH).", BODY))
story.append(Paragraph(
    "Design philosophy simple hai: URL aur email dono me alag-alag phishing signals hote hain. "
    "URL me brand impersonation, homoglyphs, suspicious TLDs, IP-based hosts jaise <i>lexical</i> cues milte hain — "
    "ye character-level CNN best capture karta hai. Email me urgency, threats, 'verify now' jaise "
    "<i>sequential language patterns</i> hote hain — ye word-level BiLSTM best capture karta hai. "
    "Fusion head dono evidence ko jointly weigh karta hai, isliye ek hi signal miss ho jaaye to bhi "
    "dusra branch cover kar leta hai.", BODY))

# ================================================================ 2. ARCHITECTURE
story.append(Paragraph("2. Model Architecture (3 Networks)", H2))

story.append(Paragraph("2.1 Branch A — URL Char-CNN (url_cnn.keras)", H3))
story.append(Paragraph(
    "Input: URL ko lowercase karke 200 characters tak kaata jaata hai, har character ek integer ID me map hota hai "
    "(0=padding, 1=unknown, 2+=vocabulary). Phir ye layers chalti hain:", BODY))
story += bullets([
    "<b>Embedding</b> (vocab ≈ 60 chars, dim 48) — har character ka dense vector",
    "<b>3 parallel Conv1D branches</b> — kernel sizes <b>2, 3, 5</b> (bi/tri/5-grams), 64 filters each, ReLU — ye n-gram patterns pakadte hain jaise 'paypal', '.tk', 'verify-login'",
    "<b>GlobalMaxPooling1D</b> per branch — sabse strong pattern signal retain hota hai",
    "<b>Concatenate</b> → <b>Dropout 0.3</b> → <b>Dense 64 (ReLU)</b> → <b>Dense 1 (sigmoid)</b> — final phishing probability",
])
story.append(Spacer(1, 3))

story.append(Paragraph("2.2 Branch B — Email BiLSTM (email_bilstm.keras)", H3))
story.append(Paragraph(
    "Input: Email body se HTML tags aur URLs strip kiye jaate hain, lowercase, whitespace-tokenize "
    "(vocab 20,000 words max), 220 tokens tak pad karte hain. Layers:", BODY))
story += bullets([
    "<b>Embedding</b> (20k vocab, dim 64)",
    "<b>SpatialDropout1D 0.2</b> — overfitting kam karne ke liye",
    "<b>Bidirectional LSTM (64 units, return_sequences=True)</b> — text dono direction me padhta hai, isliye 'URGENT … click here' jaise context patterns better capture hote hain",
    "<b>GlobalMaxPooling1D</b> → <b>Dropout 0.4</b> → <b>Dense 64 (ReLU)</b> → <b>Dense 1 (sigmoid)</b>",
])
story.append(Spacer(1, 3))

story.append(Paragraph("2.3 Head — Fusion Model (fusion_model.keras)", H3))
story.append(Paragraph(
    "Dono branches apne <b>penultimate Dense-64 layer</b> tak reuse hote hain (final sigmoid head drop karke). "
    "Dono ke 64-d feature vectors concatenate hote hain (128-d), phir:", BODY))
story += bullets([
    "<b>Dense 64 (ReLU)</b> → <b>Dropout 0.3</b> → <b>Dense 1 (sigmoid)</b>",
    "Training me branches <b>trainable</b> rehte hain (late-fusion fine-tuning), learning rate half (5e-4)",
    "Loss: <b>binary_crossentropy</b> · Optimizer: <b>Adam (1e-3 branches, 5e-4 fusion)</b>",
])
story.append(Spacer(1, 3))

arch = table([
    ["Model", "Input", "Core Layers", "Params focus"],
    ["URL char-CNN", "200 char IDs", "Embedding 48d → Conv1D (2/3/5-gram, 64f) ×3 → GMP → Dense 64 → sigmoid", "lexical patterns"],
    ["Email BiLSTM", "220 word IDs", "Embedding 64d → BiLSTM 64 → GMP → Dense 64 → sigmoid", "sequential phrasing"],
    ["Fusion", "[URL 128-d ‖ Email 64-d features]", "Concat → Dense 64 → sigmoid", "joint evidence"],
], [28*mm, 32*mm, 78*mm, 32*mm])
story.append(arch)

# ================================================================ 3. PIPELINE
story.append(Paragraph("3. End-to-End Pipeline", H2))
story.append(Paragraph(
    "<font name='Courier-Bold'>data → preprocess → train (3 models) → evaluate → artifacts → predict</font>", CODE))
story += bullets([
    "<b>1. Data loading</b> (src/data_loader.py) — data/raw/ me real CSVs (url column + label) auto-detect hote hain; na mile to deterministic <b>synthetic corpus</b> (6000 URLs, 3000 emails) generate hota hai — reproducible, CI-friendly",
    "<b>2. Preprocessing</b> (src/preprocess.py) — char-encoding URLs, from-scratch word tokenizer for emails, stratified 70/15/15 train/val/test splits (SEED=42)",
    "<b>3. Training</b> (src/train.py) — EarlyStopping (patience 4, restore best weights) + ReduceLROnPlateau (factor 0.5); max 30 epochs, batch 64",
    "<b>4. Evaluation</b> (src/evaluate.py) — accuracy, precision, recall, F1, ROC-AUC, confusion matrix; har model ke liye confusion-matrix + ROC + loss/accuracy curves artifacts/plots/ me save",
    "<b>5. Artifacts</b> — url_cnn.keras, email_bilstm.keras, fusion_model.keras, email_preprocessor.json, metrics.json, history_*.json",
    "<b>6. Inference</b> (src/predictor.py) — Predictor wrapper teeno models load karke classify(url, email) deta hai",
    "<b>7. Demo</b> — Flask app (app.py): POST /api/predict + GET /api/health + web UI (templates/index.html)",
])

# ================================================================ 4. TECH STACK
story.append(Paragraph("4. Tech Stack (Exact Versions)", H2))
story.append(Paragraph("Core stack (requirements.txt + installed):", BODY))
stack = table([
    ["Layer", "Technology", "Version", "Role"],
    ["Language", "Python", "3.13.7", "Poora codebase"],
    ["Deep Learning", "TensorFlow (CPU) + Keras 3", "2.21.0 / 3.15.1", "Model definition, training, inference"],
    ["Data handling", "pandas", "3.0.6", "CSV loading, dataframe ops"],
    ["Arrays", "NumPy", "2.5.3", "Encoding arrays, numeric ops"],
    ["ML utilities", "scikit-learn", "1.9.1", "train_test_split, metrics (precision/recall/AUC)"],
    ["Plots", "matplotlib", "3.11.2", "Confusion matrix, ROC, training curves"],
    ["Web API", "Flask + Werkzeug + Jinja2", "3.1.3", "REST endpoints + demo UI"],
    ["SciPy", "scipy", "1.18.1", "TensorFlow dependency"],
    ["Model IO", "h5py + protobuf", "3.14.0 / 7.36.2", ".keras model serialization"],
    ["Progress", "tqdm", "4.70.1", "Training progress bars"],
    ["Testing", "pytest", "9.1.1", "tests/ suite (preprocess + models)"],
    ["PDF (this report)", "reportlab", "5.0.1", "Report generation"],
], [32*mm, 46*mm, 26*mm, 66*mm], font=8.5)
story.append(stack)
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<b>Environment:</b> Windows + Python venv (.venv). Har component ka role clear hai — "
    "TensorFlow sirf modeling, sklearn sirf splits/metrics, Flask sirf serving.", BODY))

# ================================================================ 5. RESULTS
story.append(Paragraph("5. Trained Model Results (Test Set)", H2))
metrics = json.loads((config.ARTIFACTS_DIR / "metrics.json").read_text())
rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "n"]]
for name, label in [("url_cnn", "URL char-CNN"), ("email_bilstm", "Email BiLSTM"), ("fusion", "Fusion")]:
    m = metrics[name]
    rows.append([label,
                 f"{m['accuracy']:.4f}", f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                 f"{m['f1']:.4f}",
                 f"{m['roc_auc']:.4f}" if m.get("roc_auc") is not None else "—",
                 str(m["n"])])
story.append(table(rows, [34*mm, 22*mm, 22*mm, 22*mm, 22*mm, 22*mm, 18*mm], font=9))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<b>Important note:</b> Ye scores <b>synthetic corpus</b> ke test split pe hain — publication-grade nahi, "
    "pipeline validate karne ke liye hain. Real data (Kaggle Phishing Websites, SMS Spam Collection, "
    "Nazario corpus, PhishTank) data/raw/ me daal ke retrain karo to real-world numbers milenge.", BODY))

story.append(Spacer(1, 4))
story.append(Paragraph("Live API verification (run on 127.0.0.1:5000):", BODY))
live = table([
    ["Input", "URL prob", "Email prob", "Fusion", "Verdict"],
    ["paypal-secure.verify-login.tk + urgent email", "0.9879", "0.1037", "0.5867", "PHISHING (MEDIUM)"],
    ["secure-chase-online.verify-account.tk + URGENT email", "0.9838", "0.9898", "0.9989", "PHISHING (HIGH)"],
    ["github.com + normal meeting email", "0.0620", "0.0742", "0.0018", "LEGITIMATE (LOW)"],
], [72*mm, 20*mm, 22*mm, 18*mm, 38*mm], font=8.5)
story.append(live)

story.append(PageBreak())

# ================================================================ 6. PLOTS
story.append(Paragraph("6. Evaluation Plots", H2))
if (PLOTS / "url_confusion.png").exists():
    story.append(Paragraph("6.1 URL Char-CNN", H3))
    story.append(img("url_confusion.png"))
    story.append(img("url_roc.png"))
    story.append(Spacer(1, 6))
if (PLOTS / "email_confusion.png").exists():
    story.append(Paragraph("6.2 Email BiLSTM", H3))
    story.append(img("email_confusion.png"))
    story.append(img("email_roc.png"))
    story.append(Spacer(1, 6))
if (PLOTS / "fusion_confusion.png").exists():
    story.append(Paragraph("6.3 Fusion Model", H3))
    story.append(img("fusion_confusion.png"))
    story.append(img("fusion_roc.png"))
story.append(PageBreak())

# ================================================================ 7. USAGE
story.append(Paragraph("7. How to Use", H2))
story.append(Paragraph("CLI commands:", BODY))
story.append(Paragraph(
    "python main.py train              # full pipeline (data → train → evaluate)\n"
    "python main.py train --quick      # fast smoke run (~2 min)\n"
    "python main.py predict --url \"http://suspicious.tk/login\" --email \"URGENT: verify now...\"\n"
    "python main.py url --url \"http://example.com\"\n"
    "python main.py email --text \"Dear customer...\"\n"
    "python main.py app                # Flask demo → http://127.0.0.1:5000", CODE))
story.append(Paragraph("REST API:", BODY))
story.append(Paragraph(
    'POST /api/predict   {"url": "...", "email": "..."}\n'
    '  → {"phishing_probability": 0.xx, "url_probability": 0.xx,\n'
    '     "email_probability": 0.xx, "is_phishing": true/false,\n'
    '     "risk_level": "HIGH/MEDIUM/LOW"}\n\n'
    "GET  /api/health   → artifact readiness", CODE))
story.append(Paragraph("Risk bands (config.py):", BODY))
bands = table([
    ["Fusion probability", "Risk level", "Meaning"],
    ["≥ 0.75", "HIGH", "Almost certainly phishing"],
    ["0.45 – 0.75", "MEDIUM", "Suspicious — manual review recommended"],
    ["< 0.45", "LOW", "Likely legitimate"],
], [40*mm, 30*mm, 100*mm], font=9)
story.append(bands)
story.append(Spacer(1, 4))
story.append(Paragraph("Project structure:", BODY))
story.append(Paragraph(
    "config.py                  hyperparameters & paths\n"
    "main.py                    CLI entry (train/predict/url/email/app)\n"
    "app.py                     Flask demo (API + UI)\n"
    "generate_report.py         this PDF generator\n"
    "src/\n"
    "  data_loader.py           real-CSV loading + synthetic fallback\n"
    "  synth_data.py            deterministic synthetic corpus\n"
    "  preprocess.py            char encoding, tokenization, splits\n"
    "  models.py                URL CNN, email BiLSTM, fusion architecture\n"
    "  train.py                 training loops + callbacks\n"
    "  evaluate.py              metrics + confusion/ROC plots\n"
    "  predictor.py             inference wrapper\n"
    "  pipeline.py              end-to-end orchestration\n"
    "templates/index.html       demo web UI\n"
    "tests/                     pytest suite\n"
    "artifacts/                 trained models, metrics, plots", CODE))

# ================================================================ 8. LIMITATIONS
story.append(Paragraph("8. Limitations & Next Steps", H2))
story += bullets([
    "Synthetic training data — real-world generalization untested; real datasets plug karo data/raw/ me",
    "URL CNN sirf pehle 200 characters dekhta hai — lambi query strings truncate ho jaati hain",
    "Email branch ke URLs body se strip ho jaate hain (clean_email_text) — URL branch unhe nahi dekhta",
    "Threshold 0.5 fixed hai — deployment me precision/recall tradeoff tune karna hoga",
    "Possible upgrades: attention layer in BiLSTM, transformer embeddings (BERT), SHAP/LIME explainability, batch API, prediction-history database",
])

# ---------------------------------------------------------------- build
def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#8595a6"))
    canvas.drawString(18*mm, 12*mm, "Deep Learning Phishing Detection System — Technical Report")
    canvas.drawRightString(A4[0] - 18*mm, 12*mm, f"Page {doc.page}")
    canvas.restoreState()


OUT.parent.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                        leftMargin=18*mm, rightMargin=18*mm,
                        topMargin=16*mm, bottomMargin=18*mm,
                        title="Deep Learning Phishing Detection System — Technical Report")
doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
print(f"PDF written -> {OUT}")
