import numpy as np
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import cas4gnn_batch as c4g


def test_rank_and_measure_properties():
    np.random.seed(0)
    pen_grid = np.random.randn(6, 3)
    unl_pos = np.arange(6)
    m_inc = 4

    picks, r = c4g.cas_select(unl_pos, pen_grid, m_inc)

    K = pen_grid.shape[0]
    U, S, _ = np.linalg.svd(pen_grid / np.sqrt(K), full_matrices=False)
    exp_r = int(np.sum(S / S[0] > c4g.EPS_RANK)) if S[0] > c4g.EPS_RANK else 0
    assert r == exp_r
    if r > 0:
        k, s = divmod(m_inc, r)
        assert k == m_inc // r
        assert s == m_inc - k * r
        mu = np.abs(U[:, :r]) ** 2
        assert np.allclose(mu.sum(axis=0), np.ones(r))
