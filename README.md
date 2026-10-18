# News classifier: scikit-learn baseline vs TensorFlow, tracked with MLflow

Classifies 20 Newsgroups posts into 20 topics. A TF-IDF + logistic regression baseline (scikit-learn) is compared with a small
Transformer-encoder classifier (TensorFlow/Keras). Every run is tracked in MLflow, and `results.md` is generated from those
runs, so the numbers in this repo always come from a recorded run.

This README deliberately contains no results. Run the commands below and read `results.md`.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/baseline_sklearn.py        # TF-IDF + logistic regression, C chosen on the validation split
python src/train_tf.py                # Keras Transformer encoder, early stopping on validation accuracy
python src/compare.py                 # writes results.md from the latest MLflow run of each model
mlflow ui --backend-store-uri sqlite:///mlflow.db     # browse params, metrics, artifacts

python -m pytest tests                # unit tests
```

Add `--limit 2000` to either training script for a quick smoke run (a stratified subsample; not a result).
TensorFlow has no GPU support on native Windows (>= 2.11); the model is small enough to train on CPU.

## What is in each run

Parameters (data sizes, hyperparameters, library versions), per-epoch metrics for the TensorFlow model, test accuracy and
macro-F1, a per-class report, a confusion-matrix image, the most frequent wrong class pairs, and the saved model and vocabulary.

## Design decisions

- **Headers, footers and quoted replies are removed** (`fetch_20newsgroups(remove=...)`). Otherwise models can score well by
  reading sender addresses and signatures instead of the text.
- **Stratified 70/15/15 split with a fixed seed.** Hyperparameters are chosen on the validation split; the test split is scored once.
- **Macro-F1 next to accuracy**, because accuracy hides weak classes.
- **Masked mean pooling.** Many posts are empty once headers and quotes are stripped. An early version averaged over padding
  and produced `nan` loss for those posts; the pooling layer now divides by `max(real_token_count, 1)`, and
  `tests/test_smoke.py` covers an all-padding input.
- **Same tokenisation budget for both models** (about 20k vocabulary, 256 tokens) so the comparison is about the model, not the preprocessing.

## Limitations

- One seed per model, so small differences are not meaningful.
- The TensorFlow model is deliberately small (one encoder block) and lightly tuned.
- 20 Newsgroups is an old, clean benchmark; real support or news text is messier.
- No pretrained embeddings; the comparison would change with them.

## Layout

```
src/data.py              loading and splits
src/baseline_sklearn.py  TF-IDF + logistic regression
src/train_tf.py          Keras Transformer encoder
src/report.py            confusion matrix and top confusions
src/compare.py           results.md from MLflow
tests/                   unit tests
```
