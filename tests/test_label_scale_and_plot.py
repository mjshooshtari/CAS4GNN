import pathlib, sys
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import cas4gnn_batch as c4g
from sklearn.preprocessing import StandardScaler


def test_scale_labels_no_leakage_and_dtype():
    labels = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    train_idx = np.array([0, 1])
    scaled = c4g.scale_labels(labels, train_idx)
    manual = (
        StandardScaler()
        .fit(labels[train_idx].reshape(-1, 1))
        .transform(labels.reshape(-1, 1))
        .astype(np.float32)
        .squeeze()
    )
    assert scaled.dtype == np.float32
    assert np.allclose(scaled, manual)
    assert torch.from_numpy(scaled).dtype == torch.float32


def test_plot_group_log_clamp(tmp_path):
    m = np.array([1e-9, 1e-9], dtype=np.float64)
    s = np.array([2e-9, 1e-9], dtype=np.float64)
    out = tmp_path / "plot.png"
    c4g.plot_group(2, {"A": (m, s)}, "Test MSE", out)
    assert out.exists()
