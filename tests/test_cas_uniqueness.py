import numpy as np
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import cas4gnn_batch as c4g


def test_global_uniqueness_within_increment():
    np.random.seed(1)
    pen_grid = np.random.randn(10, 4)
    unl = np.arange(10)
    picks, _ = c4g.cas_select(unl, pen_grid, m_inc=7)
    assert len(picks) == len(set(picks))
