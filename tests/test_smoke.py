import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from report import save_confusion, top_confusions  # noqa: E402
from train_tf import build_model  # noqa: E402


def test_model_output_shape_and_probabilities():
    model = build_model(vocab_size=100, seq_len=16, n_classes=5)
    ids = np.random.randint(0, 100, size=(4, 16))
    out = model.predict(ids, verbose=0)
    assert out.shape == (4, 5)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-4)


def test_empty_post_is_finite():
    """Regression: a post that is empty after cleaning is all padding and used to produce NaN."""
    model = build_model(vocab_size=100, seq_len=16, n_classes=3)
    out = model.predict(np.zeros((2, 16), dtype="int32"), verbose=0)
    assert np.isfinite(out).all()


def test_partial_padding_is_finite():
    model = build_model(vocab_size=100, seq_len=16, n_classes=3)
    a = np.zeros((1, 16), dtype="int32")
    a[0, :4] = [5, 6, 7, 8]
    assert np.isfinite(model.predict(a, verbose=0)).all()


def test_confusion_helpers(tmp_path):
    names = ["a", "b", "c"]
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    cm = save_confusion(y_true, y_pred, names, str(tmp_path / "cm.png"))
    assert cm.shape == (3, 3)
    assert (tmp_path / "cm.png").exists()
    top = top_confusions(cm, names, k=2)
    assert all(count > 0 for _, _, count in top)
