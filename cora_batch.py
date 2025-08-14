#!/usr/bin/env python
# cora_batch.py
#
# CORA node‑classification · CAS vs MC · 27 GCN configs × 5 seeds
# Plots: accuracy‑vs‑samples & rank‑vs‑samples
# --------------------------------------------------------------

import argparse
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv
import sklearn.metrics as skm
import scipy.linalg as sla

# ─── experiment grid identical to synthetic script ───────────
DEPTH2 = [[8, 8], [16, 16], [32, 32]]
DEPTH3 = [[8, 16, 8], [16, 32, 16], [32, 64, 32]]
DEPTH4 = [[8, 16, 16, 8], [16, 32, 32, 16], [32, 64, 64, 32]]
WIDTHS = {2: DEPTH2, 3: DEPTH3, 4: DEPTH4}
ACTS = {"relu": torch.relu, "tanh": torch.tanh, "elu": torch.nn.functional.elu}
SEEDS = list(range(5))
M0, INC, ROUNDS = 140, 140, 5  # sample counts: 20…120
EPS_RANK = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_FILE = "Experiment.log"

parser = argparse.ArgumentParser(description="CORA node classification with CAS vs MC")
parser.add_argument("--rounds", type=int, default=ROUNDS, help="Active learning rounds")
parser.add_argument("--m0", type=int, default=M0, help="Initial labeled samples")
parser.add_argument("--inc", type=int, default=INC, help="Samples added per round")
parser.add_argument(
    "--seed",
    type=int,
    nargs="+",
    default=SEEDS,
    help="Random seeds",
)
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
    help="GCN depths to evaluate",
)
parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
parser.add_argument(
    "--smoke", action="store_true", help="Run tiny, fast settings for CI"
)


# ─── tiny helpers ─────────────────────────────────────────────
def accuracy(pred, target):
    return (pred == target).sum().item() / target.size(0)


def gcn(in_feats, hidden, out_feats, act):
    class _GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.act = act
            h = [in_feats] + hidden
            self.convs = torch.nn.ModuleList(
                GCNConv(h[i], h[i + 1]) for i in range(len(hidden))
            )
            self.out = GCNConv(h[-1], out_feats)

        def forward(self, x, e):
            for c in self.convs:
                x = self.act(c(x, e))
            pen = x.clone()
            return self.out(x, e), pen

    return _GCN()


def cas_select(unl_pos, pen_grid, m_inc):
    K, _ = pen_grid.shape
    U, S, _ = sla.svd(pen_grid / np.sqrt(K), full_matrices=False)
    if S[0] <= EPS_RANK:
        return np.random.choice(unl_pos, m_inc, replace=False), 0
    r = int(np.sum(S / S[0] > EPS_RANK))
    mu = np.abs(U[:, :r]) ** 2
    k, s = divmod(m_inc, r)
    picks = []
    for j in range(r):
        avail = unl_pos.copy()
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
        picks.append(np.random.choice(avail, 1, False, p)[0])
    return np.array(picks), r


if __name__ == "__main__":
    args = parser.parse_args()
    if args.smoke:
        args.depths = [2]
        args.acts = ["relu"]
        args.seed = [0]
        args.rounds = 1
        args.m0 = 20
        args.inc = 20
    if args.cpu:
        device = torch.device("cpu")
    SEEDS = args.seed
    M0, INC = args.m0, args.inc
    ROUNDS = args.rounds
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    ds = Planetoid(root="data", name="Cora", transform=NormalizeFeatures())
    data = ds[0].to(device)
    F = data.num_node_features
    C = int(data.y.max() + 1)
    test_mask = data.test_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()

    pool_idx = np.setdiff1d(np.arange(data.num_nodes), test_mask)
    VAL_FIXED = 200
    val_idx = np.random.choice(pool_idx, VAL_FIXED, replace=False)
    grid_idx = np.setdiff1d(pool_idx, val_idx)  # 1 508 nodes
    train_mask = grid_idx

    acts = {k: ACTS[k] for k in args.acts}
    for depth in args.depths:
        acc_dict, rank_dict = {}, {}
        width_list = WIDTHS[depth]
        for widths in width_list:
            for act_name, act_fn in acts.items():
                label = (
                    f"{widths[-1]}_{act_name}"
                    if depth == 2
                    else f"{widths[1]}_{act_name}"
                )
                acc_mat = np.zeros((len(SEEDS), ROUNDS + 1))
                rank_mat = np.zeros((len(SEEDS), ROUNDS))
                for s, seed in enumerate(SEEDS):
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    random.seed(seed)
                    labels = np.random.choice(train_mask, M0, replace=False)
                    unlab = np.setdiff1d(train_mask, labels)
                    nets = {
                        k: gcn(F, widths, C, act_fn).to(device) for k in ("CAS", "MC")
                    }
                    opts = {
                        k: torch.optim.Adam(nets[k].parameters(), lr=0.01) for k in nets
                    }
                    scheds = {
                        k: torch.optim.lr_scheduler.ReduceLROnPlateau(
                            opts[k], patience=10, factor=0.5, min_lr=1e-5
                        )
                        for k in nets
                    }
                    ce = torch.nn.CrossEntropyLoss()

                    for r in range(ROUNDS + 1):
                        for k in nets:
                            net, opt, sch = nets[k], opts[k], scheds[k]
                            net.train()
                            opt.zero_grad()
                            out, _ = net(data.x, data.edge_index)
                            loss = ce(
                                out[torch.tensor(labels, device=device)],
                                data.y[torch.tensor(labels, device=device)],
                            )
                            loss.backward()
                            opt.step()
                            with torch.no_grad():
                                val_logits, _ = net(data.x, data.edge_index)
                                val_logits = val_logits[
                                    torch.tensor(val_idx, device=device)
                                ]
                                val_pred = val_logits.argmax(dim=1)
                                val_acc = accuracy(
                                    val_pred,
                                    data.y[torch.tensor(val_idx, device=device)],
                                )
                                sch.step(-val_acc)
                            lr_now = sch.optimizer.param_groups[0]["lr"]
                            print(
                                f"[{label}_{k}_s{seed}_r{r}] TrainLoss {loss.item():.4f}  ValAcc {val_acc:.4f}  LR {lr_now:.5f}"
                            )

                        for k in nets:
                            net = nets[k]
                            pred_test = net(data.x, data.edge_index)[0][
                                torch.tensor(test_mask, device=device)
                            ].argmax(dim=1)
                            acc = accuracy(
                                pred_test,
                                data.y[torch.tensor(test_mask, device=device)],
                            )
                            acc_mat[s, r] = acc if k == "CAS" else acc_mat[s, r]
                            if k == "CAS":
                                f1 = skm.f1_score(
                                    data.y[test_mask].cpu(),
                                    pred_test.cpu(),
                                    average="macro",
                                )
                                with open(LOG_FILE, "a") as f:
                                    f.write(
                                        f"{label}_seed{seed}_r{r}  Acc {acc:.4f}  MacroF1 {f1:.4f}\n"
                                    )

                        if r == ROUNDS:
                            break
                        with torch.no_grad():
                            penult = nets["CAS"](data.x, data.edge_index)[1]
                        pen_grid = (
                            penult[torch.tensor(train_mask, device=device)]
                            .cpu()
                            .numpy()
                        )
                        unl_pos = np.searchsorted(train_mask, unlab)
                        new_pos, rk = cas_select(unl_pos, pen_grid, INC)
                        rank_mat[s, r] = rk
                        new_cas = train_mask[new_pos]
                        new_mc = np.random.choice(unlab, INC, replace=False)
                        labels = np.concatenate([labels, new_cas])
                        unlab = np.setdiff1d(unlab, new_cas)

                acc_dict[label] = (acc_mat.mean(0), acc_mat.std(0))
                rank_dict[label] = (rank_mat.mean(0), rank_mat.std(0))

        for metric, dct, ylabel, fname in [
            ("ACC", acc_dict, "Accuracy", f"acc_vs_samples_{depth}layer.png"),
            (
                "RANK",
                rank_dict,
                "Numerical rank r",
                f"rank_vs_samples_{depth}layer.png",
            ),
        ]:
            plt.figure(figsize=(6, 4))
            for lab, (m, s) in dct.items():
                steps = np.arange(len(m))
                x_vals = steps * INC + M0
                plt.plot(x_vals, m, label=lab, lw=2)
                plt.fill_between(x_vals, m - s, m + s, alpha=0.25)
            plt.xlabel("Labeled samples")
            plt.ylabel(ylabel)
            plt.title(f"{ylabel} – {depth}-layer GCNs")
            plt.grid(True, linestyle=":")
            plt.legend(fontsize=7, ncol=3)
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
