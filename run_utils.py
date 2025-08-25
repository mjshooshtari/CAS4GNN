from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple


def prepare_run(
    outdir: str,
    default_exp: str,
    run_name: str | None = None,
    exp_name: str | None = None,
) -> Tuple[Path, str]:
    """Create a timestamped run directory and latest symlinks.

    Parameters
    ----------
    outdir: str
        Base output directory (e.g., ``"results"``).
    default_exp: str
        Experiment namespace inferred from the script name.
    run_name: str | None
        Optional human-readable run identifier appended to the timestamp.
    exp_name: str | None
        Optional override for the experiment namespace.

    Returns
    -------
    Tuple[Path, str]
        The run directory path and resolved experiment name.
    """

    experiment = exp_name or default_exp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(outdir) / experiment
    suffix = f"_{run_name}" if run_name else ""
    run_dir = base / f"{timestamp}{suffix}"
    idx = 1
    while run_dir.exists():
        run_dir = base / f"{timestamp}{suffix}_{idx}"
        idx += 1
    run_dir.mkdir(parents=True)

    # Create/update symlinks
    try:
        exp_latest = base / "latest"
        if exp_latest.is_symlink() or exp_latest.exists():
            exp_latest.unlink()
        exp_latest.symlink_to(run_dir, target_is_directory=True)

        global_latest = Path(outdir) / "latest"
        if global_latest.is_symlink() or global_latest.exists():
            global_latest.unlink()
        global_latest.symlink_to(run_dir, target_is_directory=True)
    except OSError:
        # Some platforms (e.g., Windows without admin) may not support symlinks.
        pass

    return run_dir, experiment


def write_run_config(run_dir: Path, experiment: str, args: Any) -> None:
    """Persist CLI arguments as JSON alongside the run."""

    cfg = vars(args).copy()
    cfg["experiment"] = experiment
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
