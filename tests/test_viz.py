"""The walkthrough figure and page on the reference model."""

import matplotlib

matplotlib.use("Agg")

from crypto_interp.cli import main  # noqa: E402

RUN = "tests/fixtures/mul_p113_d24_m32_seed1"


def test_walkthrough_figure(tmp_path):
    out = tmp_path / "w.png"
    main(["figure", "walkthrough", "--run", RUN, "--a", "7", "--b", "12", "--out", str(out)])
    assert out.stat().st_size > 0


def test_view_writes_a_self_contained_page(tmp_path):
    out = tmp_path / "view.html"
    main(["view", "--run", RUN, "--out", str(out)])
    html = out.read_text()
    assert html.startswith("<!DOCTYPE html>") and "const P = {" in html
    assert not any(tag in html for tag in ("__CSS__", "__JS__", "__PAYLOAD__", "__TITLE__"))
    assert "<script src" not in html and "<link" not in html  # nothing external
