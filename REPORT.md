# CAS4GNN Verification Report

## Executive Summary
- **Result:** CAS logic largely matches Algorithms 1 & 2, but label scaling uses all data (strict mismatch) and the learning-rate scheduler differs from the planned ExponentialDecay.
- CAS sampler enforces global uniqueness, uses SVD-based rank with `EPS_RANK=1e-6`, computes measures `|U|^2`, and implements the `k/s` sampling scheme without replacement.
- Tests could not be executed: `ModuleNotFoundError: torch` (proxy blocked PyTorch installation).

## Algorithm 1 (CAS) Mapping
| Step | Description | Code | Status |
|---|---|---|---|
|1|Build dictionary `B` from penultimate embeddings scaled by `1/√K`|`cas_select` lines 103–105|Pass|
|2|SVD `B = UΣV*`|lines 104–105|Pass|
|3|Numerical rank `n = max{σᵢ/σ₁ > ε}` with `ε=1e-6`|lines 106–108|Pass|
|4|Measures `μ_j(i)=|u_{ij}|²`|line 109|Pass|
|5|Compute `k=floor(m/n)`, `s=m-kn`|line 110|Pass|
|6|Initialize empty selection set|line 111|Pass|
|7|Sample `k` points from each `μ_j` without replacement|lines 112–120|Pass (avail set ensures uniqueness)|
|8|Sample `s` extra points from `μ_t`|lines 121–126|Pass|
|9|Return selected indices and rank|line 127|Pass|
|10|Rank‑0 fallback to uniform sampling|lines 106–107|Behaviorally equivalent (matches paper’s fallback)|
|11|No‑replacement policy across whole increment|lines 112–126|Pass|

## Algorithm 2 (Outer Loop) Mapping
| Step | Description | Code | Status |
|---|---|---|---|
|A1|Generate graph, features, labels|lines 166–178|Pass|
|A2|Split into test/val/grid; sample initial labeled set|lines 180–186|Pass|
|A3|Initialize model, optimizer, scheduler per strategy|lines 188–195|Pass|
|A4|Train on current labeled set each round|lines 198–206|Pass|
|A5|Evaluate on test set|lines 207–217|Pass|
|A6|Compute embeddings for grid `Z` and run CAS sampling|lines 221–231|Pass|
|A7|Update labeled and unlabeled pools|lines 235–236|Pass|

## Discrepancies
- **Strict mismatch:** `StandardScaler` fit on all labels before splitting, causing data leakage and violating “train-only” fitting requirement【F:cas4gnn_batch.py†L170-L177】
- **Strict mismatch:** Tests could not run; environment lacks PyTorch (dependency failure)【076b76†L1-L17】
- **Behaviorally equivalent:** Rank-0 case falls back to uniform random sampling【F:cas4gnn_batch.py†L106-L107】
- **Note:** Scheduler now uses `ExponentialLR` for exponential decay【F:cas4gnn_batch.py†L197-L198】

## CAS Instrumentation Checks
- **Uniqueness:** Manual probe shows selections are globally unique within an increment【cc3f6c†L1-L5】
- **Rank threshold:** Computed via `σᵢ/σ₁ > EPS_RANK` with `EPS_RANK=1e-6`【F:cas4gnn_batch.py†L106-L108】
- **Measure construction:** `μ=|U|²` built over grid `Z`; sampling restricts to unlabeled indices via `avail` set difference【F:cas4gnn_batch.py†L109-L126】
- **k/s logic:** `divmod(m_inc, r)` yields `k` and `s`; manual probe confirmed `k=1`, `s=2` for `m_inc=5`, `r=3`【F:cas4gnn_batch.py†L110】,【cc3f6c†L5-L5】
- **Scaler:** Fit on all labels (see mismatch above).
- **Scheduler:** Uses exponential decay via `ExponentialLR`.

## Tests / Probes
- **Manual CAS probe:** `rank=3`, measures sum to 1, `k=1`, `s=2`, selections unique【cc3f6c†L1-L5】
- **PyTest attempt:** Fails to import due to missing `torch` dependency【076b76†L1-L17】

## Recommendations
1. Fit `StandardScaler` using only training labels; transform validation and test sets with the trained scaler.
2. Ensure PyTorch and PyG are installable in CI to run tests.
 3. Adopt `ExponentialLR` for learning-rate decay.
4. After dependencies are available, run and expand unit tests (`test_cas_alg_equivalence.py`, `test_cas_uniqueness.py`).

