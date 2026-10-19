"""The walkthrough figure on the reference model."""

import matplotlib

matplotlib.use("Agg")

from crypto_interp.cli import main  # noqa: E402

RUN = "tests/fixtures/mul_p113_d24_m32_seed1"


def test_walkthrough_figure(tmp_path):
    out = tmp_path / "w.png"
    main(["figure", "walkthrough", "--run", RUN, "--a", "7", "--b", "12", "--out", str(out)])
    assert out.stat().st_size > 0
