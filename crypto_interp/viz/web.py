"""Export a self-contained interactive HTML walkthrough of the learned algorithm.

The model is small enough to run its entire forward pass in the browser, so
the page embeds the weights as JSON plus a faithful JavaScript port of the
forward pass (verified on load against reference logits computed here in
PyTorch).

The page is a five-stage narrative mirroring the computation, each stage an
animated demonstration of one step of the algorithm:

  1. Look up the angles — each token's position on the K character clocks
     (W_E projected onto each character plane).
  2. Add the angles — the green hand rotates from θ_k(a) by θ_k(b) and lands
     on the stored embedding of a·b; helpers shown as double-speed copies.
  3. Clusters detect the sum — each character's neurons at their preferred
     phase; the activation bump points at θ_k(a)+θ_k(b).
  4. Characters vote cosets — χ_k cannot distinguish gcd(k, p−1) candidates;
     the strips show each character's surviving candidates and their
     single-point intersection (the approximate-CRT step).
  5. Interference — per-character logit waves sum to a spike at c = a·b,
     overlaid on the model's actual logits.

The page's HTML, CSS and JavaScript live in ``viz/assets/walkthrough.*`` and
are inlined at render time, so the exported file has no dependencies.

    crypto-interp view --run experiments/003_dmodel_sweep_p113/runs/<run> --out page.html
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np
import torch

from crypto_interp import measures
from crypto_interp.interp import Session
from crypto_interp.interp.bases import discrete_log_table
from crypto_interp.viz.walkthrough import char_plane, neuron_phases


def _arr(t) -> list:
    a = np.asarray(t.detach().double().numpy() if torch.is_tensor(t) else t)
    return np.round(a, 6).tolist()


def build_payload(S: Session) -> dict:
    p = S.ds.p
    g, dlog_map = discrete_log_table(p)
    dlog = [-1] + [dlog_map[a] for a in range(1, p)]

    K = list(measures.energy_support(S).K)
    h = measures.helpers(S, K=K)
    helpers = {str(k): int(m) for k, m in zip(h.helpers, h.helper_of, strict=True)}
    _, dom = S.per_neuron_dominant_char()

    m = S.model
    cfg = m.cfg
    blk = m.blocks[0]

    planes, clusters, basis_rows = {}, {}, {}
    for k in K:
        e1, e2 = char_plane(S, k)
        planes[str(k)] = {"e1": _arr(e1), "e2": _arr(e2)}
        idx = np.where(dom == k)[0]
        clusters[str(k)] = {
            "idx": idx.tolist(),
            "phi": _arr(neuron_phases(S, k, idx, e1, e2)) if len(idx) else [],
        }
        basis_rows[str(k)] = {
            "cos": _arr(S.basis[S.ci.cos[k]][:p]),
            "sin": _arr(S.basis[S.ci.sin[k]][:p]) if k in S.ci.sin else [0.0] * p,
        }

    # Reference logits for the in-browser forward-pass check.
    checks = []
    for a, b in ((3, 5), (7, 12), (p - 2, p - 11)):
        inp = torch.tensor([[a, b, S.ds.eq_token]], dtype=torch.long)
        with torch.no_grad():
            L = m(inp)[0, -1, :p]
        checks.append({"a": a, "b": b, "logits": _arr(L)})

    return {
        "p": p,
        "g": g,
        "eq": int(S.ds.eq_token),
        "dlog": dlog,
        "K": K,
        "helpers": helpers,
        "cfg": {
            "d_model": cfg.d_model,
            "d_mlp": cfg.d_mlp,
            "num_heads": cfg.num_heads,
            "d_head": cfg.d_head,
            "n_ctx": cfg.n_ctx,
        },
        "W_E": _arr(m.embed.W_E),
        "W_pos": _arr(m.pos_embed.W_pos),
        "W_Q": _arr(blk.attn.W_Q),
        "W_K": _arr(blk.attn.W_K),
        "W_V": _arr(blk.attn.W_V),
        "W_O": _arr(blk.attn.W_O),
        "W_in": _arr(blk.mlp.W_in),
        "b_in": _arr(blk.mlp.b_in),
        "W_out": _arr(blk.mlp.W_out),
        "b_out": _arr(blk.mlp.b_out),
        "W_U": _arr(m.unembed.W_U),
        "planes": planes,
        "clusters": clusters,
        "basis": basis_rows,
        "checks": checks,
    }


def _asset(name: str) -> str:
    return (resources.files("crypto_interp.viz") / "assets" / name).read_text()


def render(payload: dict, title: str) -> str:
    """The self-contained HTML page for a :func:`build_payload` payload."""
    page = (
        _asset("walkthrough.html")
        .replace("__CSS__", _asset("walkthrough.css"), 1)
        .replace("__JS__", _asset("walkthrough.js"), 1)
    )
    return (
        page.replace("__TITLE__", title)
        .replace("__P__", str(payload["p"]))
        .replace("__G__", str(payload["g"]))
        .replace("__NK__", str(len(payload["K"])))
        .replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    )


def export(run_dir: str | Path, out_file: str | Path) -> Path:
    """Write the walkthrough page for the latest checkpoint of ``run_dir``."""
    S = Session.from_run(run_dir)
    payload = build_payload(S)
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(payload, f"mod-{payload['p']} multiplication — {Path(run_dir).name}"))
    return out
