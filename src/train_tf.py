"""TensorFlow/Keras text classifier: one small Transformer encoder block over token + position embeddings."""
import argparse
import os
import time

import keras
import mlflow
import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.metrics import accuracy_score, classification_report, f1_score

from config import ARTIFACT_DIR, EXPERIMENT, TRACKING_URI
from data import SEED, load_splits
from report import save_confusion, top_confusions


@keras.saving.register_keras_serializable(package="newsclf")
class PositionEmbedding(layers.Layer):
    """Adds a learned position embedding to the token embeddings."""

    def __init__(self, seq_len, dim, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.dim = dim
        self.emb = layers.Embedding(seq_len, dim)

    def call(self, x):
        return x + self.emb(tf.range(self.seq_len))

    def get_config(self):
        cfg = super().get_config()
        cfg.update(seq_len=self.seq_len, dim=self.dim)
        return cfg


@keras.saving.register_keras_serializable(package="newsclf")
class MaskedMeanPool(layers.Layer):
    """Mean over real tokens only. Posts that are empty after cleaning are all padding; dividing by max(count, 1)
    keeps them finite (a zero vector) instead of 0/0 = NaN."""

    def call(self, inputs):
        x, keep = inputs
        m = tf.cast(keep, x.dtype)[..., None]
        return tf.reduce_sum(x * m, axis=1) / tf.maximum(tf.reduce_sum(m, axis=1), 1.0)


def build_model(vocab_size, seq_len, n_classes, dim=64, heads=2, ff_dim=128, dropout=0.2):
    ids = keras.Input(shape=(seq_len,), dtype="int32")
    pad_mask = layers.Lambda(lambda t: tf.not_equal(t, 0))(ids)  # id 0 is padding

    x = layers.Embedding(vocab_size, dim)(ids)
    x = PositionEmbedding(seq_len, dim)(x)

    attn = layers.MultiHeadAttention(num_heads=heads, key_dim=dim // heads)(x, x)
    x = layers.LayerNormalization()(x + layers.Dropout(dropout)(attn))
    ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(dim)])
    x = layers.LayerNormalization()(x + layers.Dropout(dropout)(ffn(x)))

    x = MaskedMeanPool()([x, pad_mask])  # average over real tokens only
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(ids, out)


class MlflowEpochLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="stratified subsample size (smoke runs only)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=20000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    keras.utils.set_random_seed(SEED)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), names = load_splits(args.limit)

    vec = layers.TextVectorization(max_tokens=args.vocab, output_sequence_length=args.seq_len)
    vec.adapt(tf.constant(X_tr))

    def ids(texts):
        return vec(tf.constant(texts)).numpy()

    I_tr, I_va, I_te = ids(X_tr), ids(X_va), ids(X_te)
    model = build_model(args.vocab, args.seq_len, len(names))
    model.compile(optimizer=keras.optimizers.Adam(args.lr), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    start = time.time()
    with mlflow.start_run(run_name="tensorflow-transformer-encoder"):
        mlflow.log_params({"model": "keras transformer encoder (1 block)", "vocab": args.vocab,
                           "seq_len": args.seq_len, "epochs_max": args.epochs, "batch_size": args.batch_size,
                           "lr": args.lr, "dim": 64, "heads": 2, "ff_dim": 128, "dropout": 0.2,
                           "tensorflow": tf.__version__, "limit": args.limit or "full",
                           "n_train": len(y_tr), "n_val": len(y_va), "n_test": len(y_te),
                           "params": model.count_params()})
        model.fit(I_tr, np.asarray(y_tr), validation_data=(I_va, np.asarray(y_va)), epochs=args.epochs,
                  batch_size=args.batch_size, verbose=2,
                  callbacks=[keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2,
                                                           restore_best_weights=True),
                             MlflowEpochLogger()])

        pred = model.predict(I_te, batch_size=256, verbose=0).argmax(axis=1)
        metrics = {"test_accuracy": accuracy_score(y_te, pred),
                   "test_macro_f1": f1_score(y_te, pred, average="macro"),
                   "train_seconds": time.time() - start}
        mlflow.log_metrics(metrics)

        with open(os.path.join(ARTIFACT_DIR, "tf_report.txt"), "w", encoding="utf-8") as f:
            f.write(classification_report(y_te, pred, target_names=names, zero_division=0))
        cm = save_confusion(y_te, pred, names, os.path.join(ARTIFACT_DIR, "tf_confusion.png"))
        with open(os.path.join(ARTIFACT_DIR, "tf_top_confusions.txt"), "w", encoding="utf-8") as f:
            for a, b, n in top_confusions(cm, names):
                f.write(f"{a} -> {b}: {n}\n")
        with open(os.path.join(ARTIFACT_DIR, "tf_vocab.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(vec.get_vocabulary()))
        model.save(os.path.join(ARTIFACT_DIR, "tf_model.keras"))
        mlflow.log_artifacts(ARTIFACT_DIR)
        print({k: round(v, 4) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
