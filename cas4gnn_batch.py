#!/usr/bin/env python
# cas4gnn_batch.py  –  synthetic regression with CAS vs MC
# --------------------------------------------------------
import os, random, numpy as np, torch, matplotlib.pyplot as plt
import networkx as nx, scipy.linalg as sla
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv

# ───── experiment grid ────────────────────────────────────
DEPTH2 = [[8, 8], [16, 16], [32, 32]]
DEPTH3 = [[8, 16, 8], [16, 32, 16], [32, 64, 32]]
DEPTH4 = [[8, 16, 16, 8], [16, 32, 32, 16], [32, 64, 64, 32]]
WIDTHS = {2: DEPTH2, 3: DEPTH3, 4: DEPTH4}

ACTS = {
    "relu": torch.relu,
    "tanh": torch.tanh,
    "elu" : torch.nn.functional.elu
}

SEEDS        = range(5)
N_NODES      = 10_000
TEST_FRAC    = 0.70
VAL_COUNT    = 500          # Option A
M0, INC      = 500, 500
ROUNDS       = 5            # 500 → 3 000 labels
EPS_RANK     = 1e-6
device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LOG_FILE     = "Experiment.log"
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

# ───── label function ────────────────────────────────────
def compute_label(f):
    return (2*f[0] + 3*f[1] + 4*f[2] +
            5*f[3] + 6*f[4] + np.sin(f[0]*f[4]))

# ───── GCN (last layer linear) ───────────────────────────
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim, act):
        super().__init__()
        self.act = act
        dims = [in_dim] + hidden
        self.convs = torch.nn.ModuleList(
            GCNConv(dims[i], dims[i+1]) for i in range(len(hidden)))
        self.out_conv = GCNConv(dims[-1], out_dim)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = self.act(conv(x, edge_index))
        pen = x.clone()
        x = self.out_conv(x, edge_index)
        return x, pen

# ───── CAS selection (dup‑free, rank‑0 fallback) ─────────
def cas_select(unl_pos, pen_grid, m_inc):
    K, _ = pen_grid.shape
    U, S, _ = sla.svd(pen_grid / np.sqrt(K), full_matrices=False)
    if S[0] <= EPS_RANK:
        return np.random.choice(unl_pos, m_inc, replace=False), 0
    r = int(np.sum(S / S[0] > EPS_RANK))
    mu = np.abs(U[:, :r])**2
    k, s = divmod(m_inc, r)
    picks = []
    for j in range(r):
        avail = unl_pos.copy()
        while len(picks) - j*k < k and avail.size:
            p = mu[avail, j]; p = np.maximum(p, 1e-12); p /= p.sum()
            sel = np.random.choice(avail, 1, False, p)[0]
            picks.append(sel); avail = avail[avail != sel]
    for t in range(s):
        avail = np.setdiff1d(unl_pos, picks)
        p = mu[avail, t]; p = np.maximum(p, 1e-12); p /= p.sum()
        picks.append(np.random.choice(avail, 1, False, p)[0])
    return np.array(picks, dtype=int), r

# ───── training routine (50 000 max epochs) ──────────────
def train_round(net, data, train_idx, val_idx, opt, sched):
    crit = torch.nn.MSELoss()
    best, patience = 1e9, 3; stagnant = 0
    for ep in range(50_000):
        net.train(); opt.zero_grad()
        pred, _ = net(data.x, data.edge_index)
        loss = crit(pred[train_idx], data.y[train_idx]); loss.backward(); opt.step()
        if (ep+1) % 5_000 == 0 or ep == 0:
            net.eval()
            with torch.no_grad():
                v = crit(net(data.x, data.edge_index)[0][val_idx], data.y[val_idx]).item()
            sched.step(v)
            if v < best - 1e-6: best, stagnant = v, 0
            else: stagnant += 1
            if stagnant >= patience: break
    return net

# ───── one config (hidden, act) over 5 seeds ─────────────
def run_setting(hidden, act_name, act_fn):
    mse_mat  = np.zeros((len(SEEDS), ROUNDS + 1))
    rank_mat = np.zeros((len(SEEDS), ROUNDS))
    for s, seed in enumerate(SEEDS):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        # build graph & data
        G = nx.random_geometric_graph(N_NODES, 0.05)
        feats = np.random.rand(N_NODES, 5).astype(np.float32)
        labels = np.array([compute_label(f) for f in feats], dtype=np.float32)
        feats_scaled = feats
        labels_scaled = StandardScaler().fit_transform(labels.reshape(-1,1))
        for i, f in enumerate(feats_scaled):
            G.nodes[i]['x'] = f
            G.nodes[i]['label'] = labels_scaled[i]
        data = from_networkx(G)
        data.x = torch.from_numpy(feats_scaled)
        data.y = torch.from_numpy(labels_scaled)
        data = data.to(device)

        # splits
        all_idx = np.arange(N_NODES)
        test_idx = np.random.choice(all_idx, int(TEST_FRAC*N_NODES), replace=False)
        rem = np.setdiff1d(all_idx, test_idx)
        val_idx = np.random.choice(rem, VAL_COUNT, replace=False)
        grid_idx = np.setdiff1d(rem, val_idx)           # |Z| = 2 500
        lab_init = np.random.choice(grid_idx, M0, replace=False)

        for strategy in ("CAS", "MC"):
            labels = lab_init.copy()
            unlab  = np.setdiff1d(grid_idx, labels)
            net = GCN(5, hidden, 1, act_fn).to(device)
            opt = torch.optim.Adam(net.parameters(), lr=0.01)
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                      opt, patience=3, factor=0.5, min_lr=1e-5)

            for r in range(ROUNDS + 1):
                net = train_round(net, data,
                                  torch.tensor(labels, device=device),
                                  torch.tensor(val_idx, device=device),
                                  opt, sch)
                mse = torch.nn.MSELoss()(
                    net(data.x, data.edge_index)[0][torch.tensor(test_idx, device=device)],
                    data.y[torch.tensor(test_idx, device=device)]).item()
                if strategy == "CAS":
                    mse_mat[s, r] = mse
                    with open(LOG_FILE,"a") as f:
                        f.write(f"{strategy}_{act_name}_{hidden}_seed{seed}_r{r}  TestMSE {mse:.4f}\n")
                # stop if last eval
                if r == ROUNDS: break

                # ----- sampling
                if strategy == "CAS":
                    with torch.no_grad():
                        pen = net(data.x, data.edge_index)[1]
                    pen_grid = pen[torch.tensor(grid_idx, device=device)].cpu().numpy()
                    unl_pos = np.searchsorted(grid_idx, unlab)
                    inc_now = min(INC, unlab.size)
                    if inc_now == 0: break
                    new_pos, rk = cas_select(unl_pos, pen_grid, inc_now)
                    new_ids = grid_idx[new_pos]
                    rank_mat[s, r] = rk
                else:   # MC
                    inc_now = min(INC, unlab.size)
                    if inc_now == 0: break
                    new_ids = np.random.choice(unlab, inc_now, replace=False)

                labels = np.concatenate([labels, new_ids])
                unlab  = np.setdiff1d(unlab, new_ids)

    mse_mean, mse_std = mse_mat.mean(0), mse_mat.std(0)
    rank_mean, rank_std = rank_mat.mean(0), rank_mat.std(0)
    return mse_mean, mse_std, rank_mean, rank_std

# ───── plot helper ────────────────────────────────────────
def plot_group(depth, dct, ylabel, fname):
    plt.figure(figsize=(6,4))
    for lab, (m, s) in dct.items():
        x = np.arange(len(m))*INC + M0
        plt.plot(x, m, label=lab, lw=2)
        plt.fill_between(x, m-s, m+s, alpha=0.25)
    if ylabel == "Test MSE": plt.yscale('log')
    plt.xlabel("Labeled samples"); plt.ylabel(ylabel)
    plt.title(f"{ylabel} – {depth}-layer GCNs")
    plt.grid(True, linestyle=':')
    plt.legend(fontsize=7, ncol=3)
    plt.tight_layout(); plt.savefig(fname); plt.close()

# ───── main sweep ────────────────────────────────────────
for depth, width_list in WIDTHS.items():
    mse_dict, rank_dict = {}, {}
    for widths in width_list:
        for act_name, act_fn in ACTS.items():
            tag = f"{widths[-1]}_{act_name}" if depth==2 else f"{widths[1]}_{act_name}"
            print(f"[{depth}-layer] {widths}  act={act_name}")
            mse_m, mse_s, r_m, r_s = run_setting(widths, act_name, act_fn)
            mse_dict[tag]  = (mse_m,  mse_s)
            rank_dict[tag] = (r_m, r_s)
    plot_group(depth, mse_dict,  "Test MSE",        f"mse_vs_samples_{depth}layer.png")
    plot_group(depth, rank_dict, "Numerical rank r",f"rank_vs_samples_{depth}layer.png")

