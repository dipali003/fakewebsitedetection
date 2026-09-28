"""Keras architectures: URL char-CNN, email BiLSTM, and the fusion model.

* ``build_url_cnn``      — 1-D CNN over character embeddings; captures
  spatial/lexical cues (brands, keywords, homoglyphs, TLDs).
* ``build_email_lstm``   — Bidirectional LSTM over word embeddings; captures
  sequential social-engineering patterns.
* ``build_fusion_model`` — joins frozen-or-finetuned branch features and a
  joint classification head.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models

import config
from src.preprocess import CHARS


def _set_seed() -> None:
    tf.keras.utils.set_random_seed(config.SEED)


def build_url_cnn(vocab_size: int = len(CHARS) + 2,
                  max_len: int = config.URL_MAX_LEN) -> tf.keras.Model:
    """Char-level CNN: Embedding → parallel conv (2,3,5)-grams → global pool."""
    _set_seed()
    inp = layers.Input(shape=(max_len,), dtype="int32", name="url_input")
    x = layers.Embedding(vocab_size, config.URL_EMB_DIM, name="url_emb")(inp)
    branches = []
    for k in (2, 3, 5):
        b = layers.Conv1D(config.URL_FILTERS // 2, k, padding="same",
                          activation="relu", name=f"url_conv{k}")(x)
        b = layers.GlobalMaxPooling1D(name=f"url_pool{k}")(b)
        branches.append(b)
    x = layers.Concatenate()(branches)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(config.FUSION_DENSE_UNITS, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", name="url_out")(x)
    model = models.Model(inp, out, name="url_char_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[_f1, "accuracy", tf.keras.metrics.Precision(name="prec"),
                 tf.keras.metrics.Recall(name="rec")],
    )
    return model


def build_email_lstm(vocab_size: int = config.EMAIL_VOCAB_SIZE,
                     max_tokens: int = config.EMAIL_MAX_TOKENS) -> tf.keras.Model:
    """Word-level Bidirectional LSTM for email bodies / SMS text."""
    _set_seed()
    inp = layers.Input(shape=(max_tokens,), dtype="int32", name="email_input")
    x = layers.Embedding(vocab_size, config.EMAIL_EMB_DIM, name="email_emb")(inp)
    x = layers.SpatialDropout1D(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(config.EMAIL_LSTM_UNITS,
                                         return_sequences=True))(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(config.FUSION_DENSE_UNITS, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", name="email_out")(x)
    model = models.Model(inp, out, name="email_bilstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[_f1, "accuracy", tf.keras.metrics.Precision(name="prec"),
                 tf.keras.metrics.Recall(name="rec")],
    )
    return model


def build_fusion_model(url_encoder: tf.keras.Model,
                       email_encoder: tf.keras.Model) -> tf.keras.Model:
    """Joint model over (url, email) pairs.

    Both encoders are reused *up to their penultimate dense layer*.  Their
    final sigmoid heads are dropped and the 64-d features concatenated into a
    shared classification head.  Branches stay trainable so the joint task
    can still adapt them (standard late-fusion fine-tuning).
    """
    _set_seed()
    url_in = url_encoder.input
    email_in = email_encoder.input

    def penultimate(model: tf.keras.Model) -> layers.Layer:
        # last layer is the 1-unit sigmoid head; the one before is the shared dense
        return model.get_layer(index=-2).output

    url_feat = penultimate(url_encoder)
    email_feat = penultimate(email_encoder)
    x = layers.Concatenate(name="fusion_concat")([url_feat, email_feat])
    x = layers.Dense(config.FUSION_DENSE_UNITS, activation="relu",
                     name="fusion_dense")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid", name="fusion_out")(x)
    model = models.Model([url_in, email_in], out, name="fusion_detector")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE * 0.5),
        loss="binary_crossentropy",
        metrics=[_f1, "accuracy", tf.keras.metrics.Precision(name="prec"),
                 tf.keras.metrics.Recall(name="rec")],
    )
    return model


# ------------------------------------------------------------------ metrics
def _f1(y_true, y_pred):
    """Stateless batch F1 from hard (threshold-0.5) predictions."""
    y_t = tf.reshape(tf.cast(y_true, tf.float32), (-1,))
    y_p = tf.cast(tf.reshape(y_pred, (-1,)) >= 0.5, tf.float32)
    tp = tf.reduce_sum(y_t * y_p)
    eps = tf.keras.backend.epsilon()
    prec = tp / (tf.reduce_sum(y_p) + eps)
    rec = tp / (tf.reduce_sum(y_t) + eps)
    return 2.0 * prec * rec / (prec + rec + eps)
