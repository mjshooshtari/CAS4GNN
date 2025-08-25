from argparse import Namespace

from cas4gnn_batch import SCHEDULES, apply_schedule, estimate_remaining


def test_schedule_presets():
    args = Namespace(schedule="S1", m0=0, inc=0, rounds=0)
    args = apply_schedule(args)
    assert (args.m0, args.inc, args.rounds) == SCHEDULES["S1"]
    remaining = estimate_remaining(
        10_000, 500, args.m0, args.inc, args.rounds, test_frac=0.70
    )
    assert 500 <= remaining <= 700

    args2 = Namespace(schedule="custom", m0=123, inc=45, rounds=6)
    args2 = apply_schedule(args2)
    assert (args2.m0, args2.inc, args2.rounds) == (123, 45, 6)

