# CAS4GNN

Benchmarks for Characteristic Active Sampling (CAS) versus Monte Carlo baselines on synthetic regression graphs and the Cora dataset.

## CAS4DL (MLP)

`cas4dl_batch.py` reproduces the synthetic CAS vs MC experiment using a feed-forward MLP instead of a GNN. The active-learning schedule and CAS mechanics mirror the GNN version, with embeddings taken from the MLP's penultimate layer.

### Usage

```bash
python cas4dl_batch.py --depths 2 --acts relu
```

Use `--smoke` for a quick CPU-only run. The script writes per-round metrics to `Experiment.log` and saves:

- `mse_vs_samples_<depth>layer.png` – Test MSE curves for CAS and MC (log scale).
- `rank_vs_samples_<depth>layer.png` – CAS-only numerical rank.
