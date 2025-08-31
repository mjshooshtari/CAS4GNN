#!/usr/bin/env python
# cas4dl_batch.py  –  synthetic regression with CAS vs MC using MLP
# ---------------------------------------------------------------
import argparse
import random
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.linalg as sla
from scipy.special import iv
import torch
from sklearn.preprocessing import StandardScaler

from run_utils import prepare_run, write_run_config


TEST_FRAC = 0.70

# ---- Graph heat filtering via Chebyshev polynomials ----
def normalized_adj(edge_index: torch.Tensor, num_nodes: int, device):
    """Return D^{-1/2} A D^{-1/2} as a sparse COO tensor (symmetrized)."""

    row, col = edge_index
    vals = torch.ones(row.numel(), device=device)
    A = torch.sparse_coo_tensor(
        torch.stack([row, col]), vals, (num_nodes, num_nodes)
    ).coalesce()
    A = (A + A.transpose(0, 1)).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp_min_(1e-8)
    dmh = deg.pow(-0.5)
    r, c = A.indices()
    v = A.values() * dmh[r] * dmh[c]
    return torch.sparse_coo_tensor(A.indices(), v, A.size(), device=device).coalesce()


@torch.no_grad()
def heat_filter_sparse(A_norm: torch.Tensor, X: torch.Tensor, t: float, K: int = 10):
    """Chebyshev approximation of heat diffusion exp(-tL) X."""

    coeffs = [np.exp(-t) * iv(0, t)] + [
        2.0 * np.exp(-t) * iv(k, t) for k in range(1, K + 1)
    ]
    coeffs = [torch.as_tensor(c, dtype=X.dtype, device=X.device) for c in coeffs]

    T0 = X
    out = coeffs[0] * T0
    if K >= 1:
        T1 = torch.sparse.mm(A_norm, X)
        out = out + coeffs[1] * T1
    for k in range(2, K + 1):
        Tk = 2.0 * torch.sparse.mm(A_norm, T1) - T0
        out = out + coeffs[k] * Tk
        T0, T1 = T1, Tk
    return out


@torch.no_grad()
def make_graph_target(
    edge_index: torch.Tensor,
    num_nodes: int,
    X: torch.Tensor,
    t1: float = 0.5,
    t2: float = 2.0,
    alpha: float = 1.0,
    beta: float = 0.5,
    noise: float = 0.0,
    K: int = 10,
):
    """Graph-aware regression target via heat kernels at two scales."""

    A_norm = normalized_adj(edge_index, num_nodes, X.device)
    H1 = heat_filter_sparse(A_norm, X, t=t1, K=K)
    H2 = heat_filter_sparse(A_norm, X, t=t2, K=K)
    w1 = torch.randn(X.size(1), device=X.device)
    w2 = torch.randn(X.size(1), device=X.device)
    h1 = H1 @ w1
    g2 = H2 @ w2
    h2 = H2 @ w1
    y = alpha * torch.tanh(h1 * g2) + beta * torch.sin(h2)
    if noise > 0.0:
        y = y + noise * torch.randn_like(y)
    return y.unsqueeze(1)


# preset schedules: "S1" and "S2" encourage adaptivity before the pool empties
SCHEDULES = {
    "S1": (300, 150, 10),  # up to 1,800 labels
    "S2": (400, 200, 8),  # up to 2,000 labels
}


def apply_schedule(args: argparse.Namespace) -> argparse.Namespace:
    """Override m0/inc/rounds if a preset schedule is requested."""

    if args.schedule in SCHEDULES:
        args.m0, args.inc, args.rounds = SCHEDULES[args.schedule]
    return args


def estimate_remaining(
    n_nodes: int,
    val_count: int,
    m0: int,
    inc: int,
    rounds: int,
    test_frac: float = TEST_FRAC,
) -> int:
    """Estimated unlabeled nodes after the active-learning budget."""

    train_pool = int(n_nodes * (1 - test_frac)) - val_count
    return max(0, train_pool - (m0 + inc * rounds))

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
VAL_COUNT = 500  # Option A
M0, INC = 500, 500
ROUNDS = 5  # 500 → 3 000 labels
T1, T2 = 0.5, 2.0
ALPHA, BETA = 1.0, 0.5
NOISE = 0.0
CHEBY_K = 10
EPS_RANK = 1e-6
CHK = 1_000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(
    description="Synthetic regression with CAS vs MC using an MLP"
)
parser.add_argument("--rounds", type=int, default=ROUNDS, help="Active learning rounds")
parser.add_argument("--m0", type=int, default=M0, help="Initial labeled samples")
parser.add_argument("--inc", type=int, default=INC, help="Samples added per round")
parser.add_argument(
    "--schedule",
    choices=("S1", "S2", "custom"),
    default="custom",
    help="Budget schedule preset",
)
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
parser.add_argument("--chk", type=int, default=CHK, help="Validation cadence")
parser.add_argument("--t1", type=float, default=T1, help="Heat diffusion scale t1")
parser.add_argument("--t2", type=float, default=T2, help="Heat diffusion scale t2")
parser.add_argument("--alpha", type=float, default=ALPHA, help="Target alpha")
parser.add_argument("--beta", type=float, default=BETA, help="Target beta")
parser.add_argument("--noise", type=float, default=NOISE, help="Target noise level")
parser.add_argument(
    "--cheby-K", type=int, default=CHEBY_K, help="Chebyshev polynomial order"
)
parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
parser.add_argument(
    "--smoke",
    action="store_true",
    help="Run tiny, fast settings for CI",
)
parser.add_argument("--outdir", type=str, default="results", help="Base output directory")
parser.add_argument("--run-name", type=str, default=None, help="Optional run name")
parser.add_argument("--exp-name", type=str, default=None, help="Experiment namespace override")

DEFAULT_EXP = Path(__file__).stem.split("_")[0].replace("batch", "")
if DEFAULT_EXP not in {"cas4gnn", "cas4dl"}:
    DEFAULT_EXP = "cas4dl"
RUN_DIR = Path(".")
LOG_FILE = RUN_DIR / "Experiment.log"
EXPERIMENT = DEFAULT_EXP
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
def train_round(
    net,
    data,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    base_lr: float = 1e-2,
    max_epochs: int = 50_000,
    chk: int = 1_000,
    es_patience: int = 10,
):
    """Train `net` with fresh optimizer and scheduler each round."""

    crit = torch.nn.MSELoss()
    opt = torch.optim.Adam(net.parameters(), lr=base_lr, weight_decay=5e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )

    best_v, best_state, bad = float("inf"), None, 0
    tr_hist, va_hist, lr_hist = [], [], []

    for ep in range(max_epochs):
        net.train()
        opt.zero_grad()
        pred, _ = net(data.x)
        tr_loss = crit(pred[train_idx], data.y[train_idx])
        tr_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if (ep == 0) or ((ep + 1) % chk == 0):
            net.eval()
            with torch.no_grad():
                out_v, _ = net(data.x)
                va_loss = crit(out_v[val_idx], data.y[val_idx]).item()
                tr_loss_val = tr_loss.item()
            sch.step(va_loss)
            tr_hist.append(tr_loss_val)
            va_hist.append(va_loss)
            lr_hist.append(sch.optimizer.param_groups[0]["lr"])

            if va_loss < best_v - 1e-6:
                best_v = va_loss
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if bad >= es_patience:
                break

    if best_state:
        net.load_state_dict(best_state)
    return net, tr_hist, va_hist, lr_hist


# ───── one config (hidden, act) over seeds ───────────────
def run_setting(hidden, act_name, act_fn):
    cas_mses, cas_ranks, mc_mses = [], [], []
    history_log = {"CAS": [], "MC": []}
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # ----- build graph & data -----
        G = nx.random_geometric_graph(N_NODES, 0.05)
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        feats = torch.rand(N_NODES, 5, dtype=torch.float32, device=device)

        with torch.no_grad():
            A_norm = normalized_adj(edge_index.to(device), N_NODES, device)
            X_t1 = heat_filter_sparse(A_norm, feats, t=T1, K=CHEBY_K)
            X_t2 = heat_filter_sparse(A_norm, feats, t=T2, K=CHEBY_K)
            y_raw = make_graph_target(
                edge_index.to(device),
                N_NODES,
                feats,
                t1=T1,
                t2=T2,
                alpha=ALPHA,
                beta=BETA,
                noise=NOISE,
                K=CHEBY_K,
            )
        X = torch.cat([feats, X_t1, X_t2], dim=1)

        # splits
        all_idx = np.arange(N_NODES)
        test_idx = np.random.choice(all_idx, int(TEST_FRAC * N_NODES), replace=False)
        rem = np.setdiff1d(all_idx, test_idx)
        val_idx = np.random.choice(rem, VAL_COUNT, replace=False)
        grid_idx = np.setdiff1d(rem, val_idx)
        lab_init = np.random.choice(grid_idx, M0, replace=False)

        y_np = y_raw.cpu().numpy()
        scaler = StandardScaler().fit(y_np[lab_init].reshape(-1, 1))
        labels_scaled = scaler.transform(y_np).astype(np.float32)

        data = SimpleNamespace(
            x=X.to(device),
            y=torch.from_numpy(labels_scaled).to(device),
        )

        for strategy in ("CAS", "MC"):
            labels = lab_init.copy()
            unlab = np.setdiff1d(grid_idx, labels)
            net = MLP(X.size(1), hidden, 1, act_fn).to(device)

            mses, ranks = [], []
            for r in range(ROUNDS + 1):
                # ----- PRE-ROUND LOGGING -----
                with torch.no_grad():
                    out, pen = net(data.x)
                    crit = torch.nn.MSELoss()
                    pre_tr = crit(
                        out[torch.tensor(labels, device=device)],
                        data.y[torch.tensor(labels, device=device)],
                    ).item()
                    pre_va = crit(
                        out[torch.tensor(val_idx, device=device)],
                        data.y[torch.tensor(val_idx, device=device)],
                    ).item()
                    pen_grid = (
                        pen[torch.tensor(grid_idx, device=device)].detach().cpu().numpy()
                    )
                    Kgrid = pen_grid.shape[0]
                    if Kgrid > 0:
                        U, S, _ = sla.svd(pen_grid / np.sqrt(Kgrid), full_matrices=False)
                        if S[0] > 0:
                            r_now = int(np.sum(S / S[0] > EPS_RANK))
                            k_now, s_now = (
                                INC // max(r_now, 1),
                                INC - (INC // max(r_now, 1)) * max(r_now, 1),
                            )
                            sigma_min = float(S.min())
                        else:
                            r_now, k_now, s_now, sigma_min = 0, 0, INC, 0.0
                    else:
                        r_now, k_now, s_now, sigma_min = 0, 0, INC, 0.0
                with open(LOG_FILE, "a") as f:
                    f.write(
                        f"[{EXPERIMENT}] [PRE] round {r}  | m={len(labels)}  "
                        f"train {pre_tr:.6f}  val {pre_va:.6f}  "
                        f"r {r_now}  k {k_now}  s {s_now}  sigma_min {sigma_min:.3e}\n"
                    )

                net, tr_hist, va_hist, lr_hist = train_round(
                    net,
                    data,
                    torch.tensor(labels, device=device),
                    torch.tensor(val_idx, device=device),
                    base_lr=1e-2,
                    max_epochs=50_000,
                    chk=CHK,
                    es_patience=10,
                )

                if len(va_hist):
                    with open(LOG_FILE, "a") as f:
                        f.write(
                            f"[{EXPERIMENT}] [POST] round {r}  last_train {tr_hist[-1]:.6f}  "
                            f"last_val {va_hist[-1]:.6f}  last_lr {lr_hist[-1]:.5f}\n"
                        )

                mse = torch.nn.MSELoss()(
                    net(data.x)[0][torch.tensor(test_idx, device=device)],
                    data.y[torch.tensor(test_idx, device=device)],
                ).item()
                mses.append(mse)
                with open(LOG_FILE, "a") as f:
                    f.write(
                        f"[{EXPERIMENT}] {strategy}_{act_name}_{hidden}_seed{seed}_r{r}  TestMSE {mse:.4f}\n"
                    )

                # SVD for sampling stats
                with torch.no_grad():
                    pen = net(data.x)[1]
                pen_grid = pen[torch.tensor(grid_idx, device=device)].cpu().numpy()
                U, S, _ = (
                    sla.svd(pen_grid / np.sqrt(len(grid_idx)), full_matrices=False)
                    if len(grid_idx) > 0
                    else (None, np.array([0.0]), None)
                )
                if S[0] > 0:
                    r_use = int(np.sum(S / S[0] > EPS_RANK))
                    k_use, s_use = (
                        INC // max(r_use, 1),
                        INC - (INC // max(r_use, 1)) * max(r_use, 1),
                    )
                    sigma_min_use = float(S.min())
                else:
                    r_use, k_use, s_use, sigma_min_use = 0, 0, INC, 0.0
                with open(LOG_FILE, "a") as f:
                    f.write(
                        f"[{EXPERIMENT}] [SAMPLE] round {r}  r {r_use}  k {k_use}  s {s_use}  "
                        f"sigma_min {sigma_min_use:.3e}\n"
                    )

                history_log[strategy].append(
                    {
                        "seed": seed,
                        "round": r,
                        "pre_val": pre_va,
                        "final_val": va_hist[-1] if va_hist else None,
                        "train_curve": tr_hist,
                        "val_curve": va_hist,
                        "lr_curve": lr_hist,
                        "sigma_min": sigma_min_use,
                        "r": r_use,
                        "k": k_use,
                        "s": s_use,
                    }
                )

                if r == ROUNDS:
                    break

                inc_now = min(INC, unlab.size)
                if inc_now == 0:
                    break
                if strategy == "CAS":
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

    return (
        cas_m_mean,
        cas_m_std,
        cas_r_mean,
        cas_r_std,
        mc_m_mean,
        mc_m_std,
        history_log,
    )


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
    args = apply_schedule(parser.parse_args())
    if args.smoke:
        args.depths = [2]
        args.acts = ["relu"]
        args.seed = [0]
        args.rounds = 1
        args.m0 = 10
        args.inc = 10
        args.n_nodes = 200
        args.val_count = 20
    remaining = estimate_remaining(
        args.n_nodes, args.val_count, args.m0, args.inc, args.rounds, TEST_FRAC
    )
    print(
        f"Budget: M0={args.m0} INC={args.inc} ROUNDS={args.rounds} → ~{remaining} unlabeled remain"
    )
    if args.cpu:
        device = torch.device("cpu")
    SEEDS = args.seed
    N_NODES = args.n_nodes
    VAL_COUNT = args.val_count
    M0, INC = args.m0, args.inc
    CHK = args.chk
    ROUNDS = args.rounds
    T1, T2 = args.t1, args.t2
    ALPHA, BETA = args.alpha, args.beta
    NOISE, CHEBY_K = args.noise, args.cheby_K
    RUN_DIR, EXPERIMENT = prepare_run(
        args.outdir, DEFAULT_EXP, args.run_name, args.exp_name
    )
    LOG_FILE = RUN_DIR / "Experiment.log"
    write_run_config(RUN_DIR, EXPERIMENT, args)
    print(f"Run directory: {RUN_DIR}")
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
                cas_m, cas_s, r_m, r_s, mc_m, mc_s, _ = run_setting(
                    widths, act_name, act_fn
                )
                mse_dict[f"{tag}_CAS"] = (cas_m, cas_s)
                mse_dict[f"{tag}_MC"] = (mc_m, mc_s)
                rank_dict[f"{tag}_CAS"] = (r_m, r_s)
        plot_group(
            depth, mse_dict, "Test MSE", RUN_DIR / f"mse_vs_samples_{depth}layer.png"
        )
        plot_group(
            depth, rank_dict, "Numerical rank r", RUN_DIR / f"rank_vs_samples_{depth}layer.png"
        )
