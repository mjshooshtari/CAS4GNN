# AGENTS.md — Guidance for Coding Agents (Codex, etc.)

This document tells agents how to set up, run, and verify work in the **CAS4GNN** repository.

## Repository Purpose
Active learning for GNNs comparing **CAS** (Characteristic/Compressed Active Sampling) vs **Monte Carlo** baselines on:
- **Synthetic regression graph** (`cas4gnn_batch.py`)
- **CORA node classification** (`cora_batch.py`)

## Tech Stack & Constraints
- Python ≥ 3.10
- PyTorch and PyTorch Geometric (PyG)
- numpy, scipy, scikit-learn, networkx, matplotlib
- GPU optional; code must run on CPU. If GPU is available, it may be used but should not be *required*.

## Environment Setup

> Agents must detect CUDA availability and install compatible wheels. Prefer CPU by default to avoid CUDA mismatch.

```bash
# Create env (adjust for conda/uv/venv as needed)
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install PyTorch (CPU by default)
pip install --upgrade pip
pip install "torch>=2.2,<3.0" --index-url https://download.pytorch.org/whl/cpu

# Install PyG (CPU wheels)
pip install "torch-geometric>=2.5,<3.0" "torch-scatter>=2.1,<3.0" "torch-sparse>=0.6,<1.0" "torch-cluster>=1.6,<2.0" "torch-spline-conv>=1.2,<2.0" -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

# Core deps
pip install numpy scipy scikit-learn networkx matplotlib
