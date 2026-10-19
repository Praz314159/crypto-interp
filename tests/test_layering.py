"""Package layering: lower layers must not import higher ones.

models, data  <-  training, runs  <-  interp  <-  analysis
"""

import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "crypto_interp"

# subpackage -> subpackages it may import from
ALLOWED = {
    "reference": set(),
    "models": set(),
    "data": {"reference"},
    "runs": {"models", "data"},
    "training": {"models", "data", "runs"},
    "interp": {"models", "data", "runs", "reference"},
    "measures": {"interp", "runs", "models", "data", "reference"},
    "figures": {"measures"},
    "theory": set(),
}


def _imported_subpackages(path: Path) -> set[str]:
    """crypto_interp subpackages imported by one module (absolute or relative)."""
    tree = ast.parse(path.read_text())
    sub = path.relative_to(PKG).parts[0]
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0 and (node.module or "").startswith("crypto_interp."):
                found.add(node.module.split(".")[1])
            elif node.level == 1:
                found.add(sub)
            elif node.level == 2 and node.module:
                found.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("crypto_interp."):
                    found.add(alias.name.split(".")[1])
    return found - {sub}


@pytest.mark.parametrize("sub", sorted(ALLOWED))
def test_layer_imports(sub):
    for path in (PKG / sub).rglob("*.py"):
        bad = _imported_subpackages(path) - ALLOWED[sub]
        assert not bad, f"{path.relative_to(PKG)} imports {sorted(bad)}"
