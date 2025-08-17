import torch
import numpy as np
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import cas4gnn_batch as c4g
from sklearn.preprocessing import StandardScaler


def dummy_train_round(net, data, train_idx, val_idx, opt, sched):
    return net


def test_scaler_uses_train_only(monkeypatch):
    monkeypatch.setattr(c4g, "train_round", dummy_train_round)
    c4g.device = torch.device("cpu")
    c4g.N_NODES = 20
    c4g.TEST_FRAC = 0.2
    c4g.VAL_COUNT = 2
    c4g.M0 = 5
    c4g.INC = 5
    c4g.ROUNDS = 0
    c4g.SEEDS = range(1)

    captured = {}

    class RecordingScaler(StandardScaler):
        def fit(self, X, y=None):
            captured["X"] = X.copy()
            return super().fit(X, y)

    monkeypatch.setattr(c4g, "StandardScaler", RecordingScaler)

    seed = 0
    np.random.seed(seed)
    feats = np.random.rand(c4g.N_NODES, 5).astype(np.float32)
    raw_labels = np.array([c4g.compute_label(f) for f in feats], dtype=np.float32)
    all_idx = np.arange(c4g.N_NODES)
    test_idx = np.random.choice(
        all_idx, int(c4g.TEST_FRAC * c4g.N_NODES), replace=False
    )
    rem = np.setdiff1d(all_idx, test_idx)
    val_idx = np.random.choice(rem, c4g.VAL_COUNT, replace=False)
    grid_idx = np.setdiff1d(rem, val_idx)
    lab_init = np.random.choice(grid_idx, c4g.M0, replace=False)
    expected_fit = raw_labels[lab_init].reshape(-1, 1)

    c4g.run_setting([8], "relu", torch.relu)

    assert np.allclose(captured["X"], expected_fit)
