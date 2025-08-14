import numpy as np
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import cas4gnn_batch as c4g


def test_cas_select_no_duplicates():
    np.random.seed(0)
    pen_grid = np.random.randn(10, 3)
    unl = np.arange(10)
    picks, _ = c4g.cas_select(unl, pen_grid, m_inc=5)
    assert len(picks) == len(set(picks))

    remaining = np.setdiff1d(unl, picks)
    picks2, _ = c4g.cas_select(remaining, pen_grid, m_inc=3)
    combined = np.concatenate([picks, picks2])
    assert len(combined) == len(np.unique(combined))
