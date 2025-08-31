# AGENTS.md — Guidance for Coding Agents

This document outlines how to work with the **CAS4GNN** repository.

## Workflow
- Base branch is `main`; open pull requests from feature branches.
- Prefer CPU execution. Detect CUDA but avoid CUDA-only features.
- Keep diffs small and focused.

## Running experiments
```bash
python cas4gnn_batch.py --smoke --cpu
python cas4dl_batch.py --smoke --cpu
python cora_batch.py --smoke --cpu
```
Always use `--smoke` for quick, CPU-safe verification.

## Results pipeline
- Runs write to `results/<experiment>/<timestamp>[_run-name]/`.
- `results/<experiment>/latest` and `results/latest` symlink to the newest run.
- Each run directory must contain `Experiment.log`, `run_config.json`, and figures `mse_vs_samples_<depth>layer.png` and `rank_vs_samples_<depth>layer.png`.

## CAS invariants
- Build `B` from penultimate embeddings and scale by `1/√K`.
- SVD with rank threshold `EPS_RANK=1e-6`.
- Measures are `|U·j|²`; compute `k, s = divmod(m_inc, r)`.
- Sample without replacement across the entire increment.
- Rank-0 fallback samples uniformly.
- MC baseline is computed and plotted in MSE curves.

## CLI & Presets
- `--schedule {S1,S2,custom}`: S1=300/150/10, S2=400/200/8.
- Heat-filter knobs: `--t1`, `--t2`, `--alpha`, `--beta`, `--noise`, `--cheby-K`.
- Validation cadence: `--chk`.
- Always pass `--smoke` for verification runs.

## Don’ts
- Do not modify algorithms or schedules unless explicitly tasked.
- No CUDA-only code.
- Keep diffs small.

## PR checklist
1. Plan the change and obtain approval.
2. Implement minimal diffs.
3. Provide smoke-run evidence (run directory tree and `Experiment.log` tail).
4. Ensure plots and logs are present.
