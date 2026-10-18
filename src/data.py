"""Data loading for the 20 Newsgroups classifier.

Headers, footers and quoted replies are removed so the models cannot cheat on metadata
(sender addresses, signatures); this is the standard way to keep the task honest.
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

SEED = 42


def load_splits(limit=None):
    """Return (train, val, test, class_names), stratified 70/15/15.

    limit: optional stratified subsample size, used for quick smoke runs only.
    """
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    X, y = data.data, data.target
    if limit:
        X, _, y, _ = train_test_split(X, y, train_size=limit, stratify=y, random_state=SEED)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), list(data.target_names)
