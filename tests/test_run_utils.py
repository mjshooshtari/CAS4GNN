def test_prepare_run_and_config(tmp_path):
    from types import SimpleNamespace
    import json
    from run_utils import prepare_run, write_run_config

    run_dir, exp = prepare_run(tmp_path, "cas4gnn", run_name="foo")
    assert exp == "cas4gnn"
    assert run_dir.parent == tmp_path / "cas4gnn"
    assert (tmp_path / "cas4gnn" / "latest").resolve() == run_dir.resolve()
    assert (tmp_path / "latest").resolve() == run_dir.resolve()

    args = SimpleNamespace(outdir=str(tmp_path), run_name="foo", exp_name=None)
    write_run_config(run_dir, exp, args)
    cfg = json.loads((run_dir / "run_config.json").read_text())
    assert cfg["experiment"] == "cas4gnn"
