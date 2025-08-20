#!/usr/bin/env python
# cas4dl_batch.py  –  synthetic regression with CAS vs MC using MLP
# ---------------------------------------------------------------
import argparse
import os
import random
from types import SimpleNamespace

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.linalg as sla
import torch
from sklearn.preprocessing import StandardScaler

# ───── experiment grid ────────────────────────────────────
DEPTH2 = [[10, 10], [20, 20], [30, 30]]
DEPTH3 = [[10, 10, 10], [20, 20, 20], [30, 30, 30]]
DEPTH4 = [[10, 10, 10, 10], [20, 20, 20, 20], [30, 30, 30, 30]]
WIDTHS = {2: DEPTH2, 3: DEPTH3, 4: DEPTH4}

ACTS = {
    "relu": torch.relu,
    "tanh": torch.tanh,
    "elu": torch.nn.functional.elu,
}

# defaults (overridable via CLI)
SEEDS = list(range(5))
N_NODES = 10_000
TEST_FRAC = 0.70
VAL_COUNT = 500  # Option A
M0, INC = 500, 500
ROUNDS = 5  # 500 → 3 000 labels
EPS_RANK = 1e-6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(
    description="Synthetic regression with CAS vs MC using an MLP"
)
parser.add_argument("--rounds", type=int, default=ROUNDS, help="Active learning rounds")
parser.add_argument("--m0", type=int, default=M0, help="Initial labeled samples")
parser.add_argument("--inc", type=int, default=INC, help="Samples added per round")
parser.add_argument("--seed", type=int, nargs="+", default=SEEDS, help="Random seeds")
parser.add_argument(
    "--acts",
    nargs="+",
    default=list(ACTS.keys()),
    help="Activation functions",
)
parser.add_argument(
    "--depths",
    type=int,
    nargs="+",
    default=list(WIDTHS.keys()),
    help="MLP depths to evaluate",
)
parser.add_argument("--n-nodes", type=int, default=N_NODES, help="Synthetic node count")
parser.add_argument("--val-count", type=int, default=VAL_COUNT, help="Validation count")
parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
parser.add_argument(
    "--smoke",
    action="store_true",
    help="Run tiny, fast settings for CI",
)

LOG_FILE = "Experiment.log"


# ───── label function ────────────────────────────────────
def compute_label(f: np.ndarray) -> float:
    return 2 * f[0] + 3 * f[1] + 4 * f[2] + 5 * f[3] + 6 * f[4] + np.sin(f[0] * f[4])


# ───── MLP (last layer linear) ───────────────────────────
class MLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int, act):
        super().__init__()
        self.act = act
        dims = [in_dim] + hidden
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden))
        )
        self.out = torch.nn.Linear(dims[-1], out_dim)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = self.act(layer(x))
        pen = x.clone()
        x = self.out(x)
        return x, pen


# ───── CAS selection (dup‑free, rank‑0 fallback) ─────────
def cas_select(unl_pos: np.ndarray, pen_grid: np.ndarray, m_inc: int):
    K, _ = pen_grid.shape
    U, S, _ = sla.svd(pen_grid / np.sqrt(K), full_matrices=False)
    if S[0] <= EPS_RANK:
        return np.random.choice(unl_pos, m_inc, replace=False), 0
    r = int(np.sum(S / S[0] > EPS_RANK))
    mu = np.abs(U[:, :r]) ** 2
    k, s = divmod(m_inc, r)
    picks = []
    for j in range(r):
        avail = np.setdiff1d(unl_pos, picks)
        while len(picks) - j * k < k and avail.size:
            p = mu[avail, j]
            p = np.maximum(p, 1e-12)
            p /= p.sum()
            sel = np.random.choice(avail, 1, False, p)[0]
            picks.append(sel)
            avail = avail[avail != sel]
    for t in range(s):
        avail = np.setdiff1d(unl_pos, picks)
        p = mu[avail, t]
        p = np.maximum(p, 1e-12)
        p /= p.sum()
        sel = np.random.choice(avail, 1, False, p)[0]
        picks.append(sel)
    return np.array(picks), r


# ───── training ───────────────────────────────────────────
def train_round(net, data, train_idx, val_idx, opt, sched):
    crit = torch.nn.MSELoss()
    best, patience = 1e9, 3
    stagnant = 0
    for ep in range(50_000):
        net.train()
        opt.zero_grad()
        pred, _ = net(data.x)
        loss = crit(pred[train_idx], data.y[train_idx])
        loss.backward()
        opt.step()
        sched.step()
        if (ep + 1) % 5_000 == 0 or ep == 0:
            net.eval()
            with torch.no_grad():
                v = crit(net(data.x)[0][val_idx], data.y[val_idx]).item()
            if v < best - 1e-6:
                best, stagnant = v, 0
            else:
                stagnant += 1
            if stagnant >= patience:
                break
    return net


# ───── one config (hidden, act) over seeds ───────────────
def run_setting(hidden, act_name, act_fn):
    cas_mses, cas_ranks, mc_mses = [], [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # build synthetic data
        G = nx.random_geometric_graph(N_NODES, 0.05)
        feats = np.random.rand(N_NODES, 5).astype(np.float32)
        raw_labels = np.array([compute_label(f) for f in feats], dtype=np.float32)
        feats_scaled = feats

        # splits
        all_idx = np.arange(N_NODES)
        test_idx = np.random.choice(all_idx, int(TEST_FRAC * N_NODES), replace=False)
        rem = np.setdiff1d(all_idx, test_idx)
        val_idx = np.random.choice(rem, VAL_COUNT, replace=False)
        grid_idx = np.setdiff1d(rem, val_idx)
        lab_init = np.random.choice(grid_idx, M0, replace=False)

        # fit scaler on training labels only, then transform full label set
        scaler = StandardScaler().fit(raw_labels[lab_init].reshape(-1, 1))
        labels_scaled = scaler.transform(raw_labels.reshape(-1, 1)).astype(np.float32)

        data = SimpleNamespace(
            x=torch.from_numpy(feats_scaled).to(device),
            y=torch.from_numpy(labels_scaled).to(device),
        )

        for strategy in ("CAS", "MC"):
            labels = lab_init.copy()
            unlab = np.setdiff1d(grid_idx, labels)
            net = MLP(5, hidden, 1, act_fn).to(device)
            opt = torch.optim.Adam(net.parameters(), lr=0.01)
            sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)

            mses, ranks = [], []
            for r in range(ROUNDS + 1):
                net = train_round(
                    net,
                    data,
                    torch.tensor(labels, device=device),
                    torch.tensor(val_idx, device=device),
                    opt,
                    sch,
                )
                mse = torch.nn.MSELoss()(
                    net(data.x)[0][torch.tensor(test_idx, device=device)],
                    data.y[torch.tensor(test_idx, device=device)],
                ).item()
                mses.append(mse)
                with open(LOG_FILE, "a") as f:
                    f.write(
                        f"{strategy}_MLP_{act_name}_{hidden}_seed{seed}_r{r}  TestMSE {mse:.4f}\n"
                    )
                if r == ROUNDS:
                    break

                inc_now = min(INC, unlab.size)
                if inc_now == 0:
                    break
                if strategy == "CAS":
                    with torch.no_grad():
                        pen = net(data.x)[1]
                    pen_grid = pen[torch.tensor(grid_idx, device=device)].cpu().numpy()
                    unl_pos = np.searchsorted(grid_idx, unlab)
                    new_pos, rk = cas_select(unl_pos, pen_grid, inc_now)
                    new_ids = grid_idx[new_pos]
                    ranks.append(rk)
                else:  # MC
                    new_ids = np.random.choice(unlab, inc_now, replace=False)

                labels = np.concatenate([labels, new_ids])
                unlab = np.setdiff1d(unlab, new_ids)

            if strategy == "CAS":
                cas_mses.append(mses)
                cas_ranks.append(ranks)
            else:
                mc_mses.append(mses)

    def _agg(arrs):
        L = min(len(a) for a in arrs)
        mat = np.array([a[:L] for a in arrs])
        return mat.mean(0), mat.std(0)

    cas_m_mean, cas_m_std = _agg(cas_mses)
    mc_m_mean, mc_m_std = _agg(mc_mses)
    cas_r_mean, cas_r_std = (
        _agg(cas_ranks) if cas_ranks else (np.array([]), np.array([]))
    )

    return cas_m_mean, cas_m_std, cas_r_mean, cas_r_std, mc_m_mean, mc_m_std


# ───── plot helper ────────────────────────────────────────
def plot_group(depth, dct, ylabel, fname):
    plt.figure(figsize=(6, 4))
    for lab, (m, s) in dct.items():
        x = np.arange(len(m)) * INC + M0
        if ylabel == "Test MSE":
            lower = np.maximum(m - s, 1e-12)
            plt.yscale("log")
            plt.fill_between(x, lower, m + s, alpha=0.25)
        else:
            plt.fill_between(x, m - s, m + s, alpha=0.25)
        plt.plot(x, m, label=lab, lw=2)
    plt.xlabel("Labeled samples")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} – {depth}-layer MLPs")
    plt.grid(True, linestyle=":")
    plt.legend(fontsize=7, ncol=3)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


if __name__ == "__main__":
    args = parser.parse_args()
    if args.smoke:
        args.depths = [2]
        args.acts = ["relu"]
        args.seed = [0]
        args.rounds = 1
        args.m0 = 10
        args.inc = 10
        args.n_nodes = 200
        args.val_count = 20
    if args.cpu:
        device = torch.device("cpu")
    SEEDS = args.seed
    N_NODES = args.n_nodes
    VAL_COUNT = args.val_count
    M0, INC = args.m0, args.inc
    ROUNDS = args.rounds
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    acts = {k: ACTS[k] for k in args.acts}
    for depth in args.depths:
        width_list = WIDTHS[depth]
        mse_dict, rank_dict = {}, {}
        for widths in width_list:
            for act_name, act_fn in acts.items():
                tag = (
                    f"{widths[-1]}_{act_name}"
                    if depth == 2
                    else f"{widths[1]}_{act_name}"
                )
                print(f"[{depth}-layer] {widths}  act={act_name}")
                cas_m, cas_s, r_m, r_s, mc_m, mc_s = run_setting(
                    widths, act_name, act_fn
                )
                mse_dict[f"{tag}_CAS"] = (cas_m, cas_s)
                mse_dict[f"{tag}_MC"] = (mc_m, mc_s)
                rank_dict[f"{tag}_CAS"] = (r_m, r_s)
        plot_group(depth, mse_dict, "Test MSE", f"mse_vs_samples_{depth}layer.png")
        plot_group(
            depth, rank_dict, "Numerical rank r", f"rank_vs_samples_{depth}layer.png"
        )
