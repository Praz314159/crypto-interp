"""The crypto-interp command, end to end on a tiny experiment."""

import pandas as pd
import pytest

from crypto_interp.cli import main
from crypto_interp.cli.select import parse_value
from crypto_interp.runs import discover

CONFIG = """
from crypto_interp.training import ExperimentConfig
CONFIG = ExperimentConfig(task="mul", p=7, d_model=8, d_mlp=16, num_heads=2, d_head=4,
                          num_epochs=12, save_every=5, log_every=1000)
"""


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("experiments")
    (root / "001_toy").mkdir()
    (root / "001_toy" / "config.py").write_text(CONFIG)
    return root


def test_parse_value():
    assert parse_value("3") == 3 and parse_value("0.5") == 0.5 and parse_value("x") == "x"
    assert parse_value("1..3") == range(1, 4) and parse_value("2,4") == [2, 4]


def test_train_refuses_an_occupied_run_directory(root):
    args = ["train", "--root", str(root), "--experiment", "001_toy", "--seed", "0"]
    main(args + ["--tag", "solo"])
    assert (root / "001_toy" / "runs" / "solo" / "checkpoint_000011.pt").exists()
    with pytest.raises(SystemExit, match="already has checkpoints"):
        main(args + ["--tag", "solo"])


def test_sweep_then_measure_and_figure(root, tmp_path, capsys):
    sweep = [
        "sweep",
        "--root",
        str(root),
        "--experiment",
        "001_toy",
        "--seeds",
        "1..2",
        "--grid",
        "d_mlp=8,16",
        "--jobs",
        "4",
        "--tag-format",
        "m{d_mlp}_seed{seed}",
    ]
    main(sweep)
    names = {r.name for r in discover(root)}
    assert {"m8_seed1", "m8_seed2", "m16_seed1", "m16_seed2"} <= names

    main(sweep + ["--skip-existing"])
    assert capsys.readouterr().out.count("(finished)") == 4

    out = tmp_path / "tables" / "status.csv"  # parent directories are created
    main(["measure", "grok_status", "--root", str(root), "--where", "d_mlp=8", "--out", str(out)])
    assert set(pd.read_csv(out)["seed"]) == {1, 2}

    png = tmp_path / "regimes.png"
    main(["figure", "regimes", "--root", str(root), "--experiment", "001", "--out", str(png)])
    assert png.stat().st_size > 0

    sel = ["--select", "d_mlp=8 seed=1", "--select", "d_mlp=16 seed=2"]
    main(["measure", "grok_status", "--root", str(root), *sel, "--out", str(out)])
    rows = pd.read_csv(out)[["d_mlp", "seed"]].itertuples(index=False)
    assert {tuple(r) for r in rows} == {(8, 1), (16, 2)}

    main(["runs", "ls", "--root", str(root)])
    assert "m16_seed2" in capsys.readouterr().out
