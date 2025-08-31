# CAS4GNN

## Summary
Characteristic Active Sampling (CAS) experiments for graph neural networks and multilayer perceptrons compare CAS against Monte Carlo (MC) baselines on synthetic regression graphs and the Cora node-classification benchmark. Runs produce metrics and plots for test mean-squared error (MSE) and numerical rank while logging summaries per round.

## Quickstart (CPU)
```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python cas4gnn_batch.py --smoke --cpu
python cas4dl_batch.py --smoke --cpu
```
To smoke-test the Cora script:
```bash
python cora_batch.py --smoke --cpu
```

## CLI reference
| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--rounds` | int | 5 | Active learning rounds |
| `--m0` | int | 500 | Initial labeled samples |
| `--inc` | int | 500 | Samples added per round |
| `--schedule` | str | custom | Budget preset (S1:300/150/10, S2:400/200/8) |
| `--seed` | int list | 0 1 2 3 4 | Random seeds |
| `--acts` | str list | relu tanh elu | Activation functions |
| `--depths` | int list | 2 3 4 | Model depths |
| `--n-nodes` | int | 10000 | Synthetic node count |
| `--val-count` | int | 500 | Validation node count |
| `--chk` | int | 1000 | Validation cadence |
| `--t1` | float | 0.5 | Heat diffusion scale t1 |
| `--t2` | float | 2.0 | Heat diffusion scale t2 |
| `--alpha` | float | 1.0 | Target alpha |
| `--beta` | float | 0.5 | Target beta |
| `--noise` | float | 0.0 | Target noise level |
| `--cheby-K` | int | 10 | Chebyshev polynomial order |
| `--cpu` | flag | False | Force CPU execution |
| `--smoke` | flag | False | Tiny profile for quick checks |
| `--outdir` | str | results | Base output directory |
| `--run-name` | str | None | Optional run identifier |
| `--exp-name` | str | None | Experiment namespace override |

## Results & artifacts
- Runs are saved under `results/<experiment>/<timestamp>[_run-name]/`.
- `results/<experiment>/latest` and `results/latest` point to the most recent run.
- Each run directory contains `Experiment.log`, `run_config.json`, and per-depth figures `mse_vs_samples_<depth>layer.png` and `rank_vs_samples_<depth>layer.png`.
- The run's `Experiment.log` logs one-line summaries per round.

## Figures generated
- `mse_vs_samples_<depth>layer.png` – CAS and MC test MSE vs. labeled samples.
- `rank_vs_samples_<depth>layer.png` – CAS numerical rank vs. labeled samples.

## Reproducibility
- Random seeds controlled via `--seed`; dtype is float32.
- Works on CPU or GPU; smoke runs complete in seconds, full runs take minutes per depth.
- For deterministic CPU runs, pass `--cpu`.

## CI
GitHub Actions installs CPU wheels and runs `pytest` on pushes and pull requests.

## Contributing
1. Branch from `main` and keep diffs small.
2. Document intent and include smoke-run evidence.
3. Open a pull request when tests pass.

## License
This project is licensed under the terms of the [LICENSE](LICENSE).
