import torch
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import cas4gnn_batch as c4g


def dummy_train_round(net, data, train_idx, val_idx, opt, sched):
    return net


def test_metrics_shapes(monkeypatch):
    monkeypatch.setattr(c4g, "train_round", dummy_train_round)
    c4g.device = torch.device("cpu")
    c4g.N_NODES = 20
    c4g.TEST_FRAC = 0.2
    c4g.VAL_COUNT = 2
    c4g.M0 = 5
    c4g.INC = 5
    c4g.ROUNDS = 5
    c4g.SEEDS = range(1)

    cas_m, cas_s, r_m, r_s, mc_m, mc_s = c4g.run_setting([8, 8], "relu", torch.relu)

    assert cas_m.size == cas_s.size
    assert mc_m.size == mc_s.size

    grid_size = int((1 - c4g.TEST_FRAC) * c4g.N_NODES) - c4g.VAL_COUNT
    unlab_initial = grid_size - c4g.M0
    expected_inc = (unlab_initial + c4g.INC - 1) // c4g.INC if unlab_initial > 0 else 0
    expected_rounds = expected_inc + 1

    assert cas_m.shape[0] == mc_m.shape[0] == expected_rounds
    assert r_m.shape[0] == expected_inc
