import torch
import numpy as np
import pathlib
import sys
import random
import networkx as nx
from torch_geometric.utils import from_networkx

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
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    G = nx.random_geometric_graph(c4g.N_NODES, 0.05)
    feats = np.random.rand(c4g.N_NODES, 5).astype(np.float32)
    for i in range(c4g.N_NODES):
        G.nodes[i]["x"] = feats[i]
    data = from_networkx(G)
    X = torch.from_numpy(feats)
    edge_index = data.edge_index
    with torch.no_grad():
        y_raw = c4g.make_graph_target(
            edge_index,
            c4g.N_NODES,
            X,
            t1=c4g.T1,
            t2=c4g.T2,
            alpha=c4g.ALPHA,
            beta=c4g.BETA,
            noise=c4g.NOISE,
            K=c4g.CHEBY_K,
        ).cpu().numpy()
    all_idx = np.arange(c4g.N_NODES)
    test_idx = np.random.choice(
        all_idx, int(c4g.TEST_FRAC * c4g.N_NODES), replace=False
    )
    rem = np.setdiff1d(all_idx, test_idx)
    val_idx = np.random.choice(rem, c4g.VAL_COUNT, replace=False)
    grid_idx = np.setdiff1d(rem, val_idx)
    lab_init = np.random.choice(grid_idx, c4g.M0, replace=False)
    expected_fit = y_raw[lab_init]

    c4g.run_setting([8], "relu", torch.relu)

    assert np.allclose(captured["X"], expected_fit)
