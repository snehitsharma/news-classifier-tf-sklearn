"""Classical baseline: TF-IDF + logistic regression, tuned on the validation split, scored once on test."""
import argparse
import os
import time

import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from config import ARTIFACT_DIR, EXPERIMENT, TRACKING_URI
from data import load_splits
from report import save_confusion, top_confusions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="stratified subsample size (smoke runs only)")
    ap.add_argument("--c-values", type=float, nargs="+", default=[1.0, 10.0, 30.0])
    args = ap.parse_args()

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), names = load_splits(args.limit)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    start = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)
    A_tr = vec.fit_transform(X_tr)
    A_va, A_te = vec.transform(X_va), vec.transform(X_te)

    with mlflow.start_run(run_name="sklearn-tfidf-logreg"):
        mlflow.log_params({"model": "tfidf+logreg", "ngram_range": "1-2", "min_df": 2, "max_features": 100000,
                           "c_values": str(args.c_values), "limit": args.limit or "full",
                           "n_train": len(y_tr), "n_val": len(y_va), "n_test": len(y_te)})
        best = None
        for c in args.c_values:
            clf = LogisticRegression(C=c, max_iter=1000)
            clf.fit(A_tr, y_tr)
            val_f1 = f1_score(y_va, clf.predict(A_va), average="macro")
            mlflow.log_metric(f"val_macro_f1_C{c:g}", val_f1)
            if best is None or val_f1 > best[0]:
                best = (val_f1, c, clf)
        val_f1, c, clf = best
        pred = clf.predict(A_te)
        metrics = {"test_accuracy": accuracy_score(y_te, pred),
                   "test_macro_f1": f1_score(y_te, pred, average="macro"),
                   "val_macro_f1": val_f1, "train_seconds": time.time() - start}
        mlflow.log_param("best_C", c)
        mlflow.log_metrics(metrics)

        report_path = os.path.join(ARTIFACT_DIR, "sklearn_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(classification_report(y_te, pred, target_names=names, zero_division=0))
        cm = save_confusion(y_te, pred, names, os.path.join(ARTIFACT_DIR, "sklearn_confusion.png"))
        with open(os.path.join(ARTIFACT_DIR, "sklearn_top_confusions.txt"), "w", encoding="utf-8") as f:
            for a, b, n in top_confusions(cm, names):
                f.write(f"{a} -> {b}: {n}\n")
        mlflow.log_artifacts(ARTIFACT_DIR)
        print({k: round(v, 4) for k, v in metrics.items()}, "best C =", c)


if __name__ == "__main__":
    main()
