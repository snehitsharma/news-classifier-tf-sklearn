"""Read the latest run of each model from MLflow and write results.md (the only source of numbers for the README)."""
import mlflow

from config import EXPERIMENT, TRACKING_URI


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT], order_by=["start_time DESC"])
    if runs.empty:
        raise SystemExit("No runs yet. Run baseline_sklearn.py and train_tf.py first.")
    latest = runs.drop_duplicates(subset="tags.mlflow.runName", keep="first")
    lines = ["| Model | Test accuracy | Test macro-F1 | Train seconds | Train/val/test sizes |", "|---|---|---|---|---|"]
    for _, r in latest.iterrows():
        sizes = "/".join(str(r.get(f"params.n_{s}", "?")) for s in ("train", "val", "test"))
        lines.append(f"| {r['tags.mlflow.runName']} | {r['metrics.test_accuracy']:.4f} | "
                     f"{r['metrics.test_macro_f1']:.4f} | {r['metrics.train_seconds']:.0f} | {sizes} |")
    text = "\n".join(lines) + "\n"
    open("results.md", "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
