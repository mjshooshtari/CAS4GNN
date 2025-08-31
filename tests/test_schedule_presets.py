from argparse import Namespace

import pytest

try:  # gnn script requires torch-geometric
    import cas4gnn_batch as gnn  # type: ignore
except Exception:  # pragma: no cover - dependency not installed
    gnn = None
import cas4dl_batch as dl

MODS = [dl] if gnn is None else [gnn, dl]


@pytest.mark.parametrize("mod", MODS)
def test_schedule_presets(mod):
    args = Namespace(schedule="S1", m0=0, inc=0, rounds=0)
    args = mod.apply_schedule(args)
    assert (args.m0, args.inc, args.rounds) == mod.SCHEDULES["S1"]
    remaining = mod.estimate_remaining(
        10_000, 500, args.m0, args.inc, args.rounds, test_frac=0.70
    )
    assert 500 <= remaining <= 700

    args2 = Namespace(schedule="custom", m0=123, inc=45, rounds=6)
    args2 = mod.apply_schedule(args2)
    assert (args2.m0, args2.inc, args2.rounds) == (123, 45, 6)

