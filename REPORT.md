# CAS4GNN Verification Report

## Executive Summary
CAS4GNN and CAS4DL implement Characteristic Active Sampling (CAS) against Monte Carlo baselines on synthetic graphs and a feed-forward MLP. The code enforces CAS invariants and logs results through a unified pipeline; assumptions on scaling, scheduling, and sampling were verified.

## Algorithm-to-Code Mapping
### Algorithm 1 – CAS sampler
| Step | Code | Status |
| --- | --- | --- |
| Build `B` from penultimate embeddings, scale by `1/√K` | `cas4gnn_batch.py` lines 222‑245 | Pass |
| SVD and rank `r = Σᵢ/Σ₁ > 1e-6` | `cas4gnn_batch.py` lines 222‑245 | Pass |
| Measures `|U·j|²` and `k,s = divmod(m_inc,r)` | `cas4gnn_batch.py` lines 222‑245 | Pass |
| Sample `k` per measure and `s` extras without replacement | `cas4gnn_batch.py` lines 222‑245 | Pass |
| Rank‑0 → uniform sampling | `cas4gnn_batch.py` lines 225‑229 | Behaviorally equivalent |

### Algorithm 2 – Outer loop
| Step | Code | Status |
| --- | --- | --- |
| Generate graph, splits, and initial labeled set | `cas4gnn_batch.py` lines 338‑357 | Pass |
| Train with LR scheduler, clipping, chk cadence, early stop | `cas4gnn_batch.py` lines 248‑308, 392‑433 | Pass |
| CAS sampling and pool update | `cas4gnn_batch.py` lines 459‑485 | Pass |
| Log metrics and plots | `cas4gnn_batch.py` lines 392‑456, 524‑541 | Pass |

## Implementation Notes
- StandardScaler fit on training labels only.
- `ReduceLROnPlateau` scheduler with representation-active cadence.
- Gradients clipped to 1.0.
- Validation cadence set by `--chk`; early-stop patience of 10.

## Results Pipeline & Plots
`run_utils.py` creates `results/<experiment>/<timestamp>[_run-name]/` with per-experiment and global `latest` symlinks and writes `run_config.json`. Each run produces `Experiment.log` plus `mse_vs_samples_<depth>layer.png` and `rank_vs_samples_<depth>layer.png`.

## Open Questions / Future Work
- Extend experiments to additional datasets.
- Expand unit tests for results and plotting utilities.
