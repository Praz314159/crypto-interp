"""Export a self-contained interactive HTML viewer of the learned algorithm.

The model is small enough to run its entire forward pass in the browser, so
the page embeds the weights as JSON plus a faithful JavaScript port of the
forward pass (verified on load against reference logits computed here in
PyTorch). Sliders pick (a, b); every panel recomputes live:

  - clocks: token embeddings in each essential character's 2D plane of W_E,
    with hands at a, b, and a·b
  - attention pattern at the '=' position
  - phased arrays: each character cluster's neurons at their preferred phase,
    bar length = actual activation; the bump points at θ_k(a) + θ_k(b)
  - interference: the live logits decomposed per character over candidates in
    dlog order; the per-character waves are coset-ambiguous, their sum spikes
    at c = a·b
  - "animate" steps b ← g·b so multiplication becomes uniform rotation

Usage:
    python -m crypto_interp.analysis.inference_viewer_web \\
        --run-dir experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_20_wd2_seed1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from crypto_interp.interp import Session
from crypto_interp.interp.bases import discrete_log_table
from crypto_interp.analysis.inference_viewer import char_plane, neuron_phases


def _arr(t) -> list:
    a = np.asarray(t.detach().double().numpy() if torch.is_tensor(t) else t)
    return np.round(a, 6).tolist()


def build_payload(S: Session) -> dict:
    p = S.ds.p
    g, dlog_map = discrete_log_table(p)
    dlog = [-1] + [dlog_map[a] for a in range(1, p)]

    K = sorted(int(k) for k in S.essential()["K"])
    helpers = {str(h): int(m) for (h, m, _mult, _e) in S.helpers(K)}
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
            "sin": _arr(S.basis[S.ci.sin[k]][:p]) if k in S.ci.sin
                   else [0.0] * p,
        }

    # Reference logits for the in-browser forward-pass check.
    checks = []
    for a, b in ((3, 5), (7, 12), (p - 2, p - 11)):
        inp = torch.tensor([[a, b, S.ds.eq_token]], dtype=torch.long)
        with torch.no_grad():
            L = m(inp)[0, -1, :p]
        checks.append({"a": a, "b": b, "logits": _arr(L)})

    return {
        "p": p, "g": g, "eq": int(S.ds.eq_token), "dlog": dlog,
        "K": K, "helpers": helpers,
        "cfg": {"d_model": cfg.d_model, "d_mlp": cfg.d_mlp,
                "num_heads": cfg.num_heads, "d_head": cfg.d_head,
                "n_ctx": cfg.n_ctx},
        "W_E": _arr(m.embed.W_E), "W_pos": _arr(m.pos_embed.W_pos),
        "W_Q": _arr(blk.attn.W_Q), "W_K": _arr(blk.attn.W_K),
        "W_V": _arr(blk.attn.W_V), "W_O": _arr(blk.attn.W_O),
        "W_in": _arr(blk.mlp.W_in), "b_in": _arr(blk.mlp.b_in),
        "W_out": _arr(blk.mlp.W_out), "b_out": _arr(blk.mlp.b_out),
        "W_U": _arr(m.unembed.W_U),
        "planes": planes, "clusters": clusters, "basis": basis_rows,
        "checks": checks,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  body { font-family: -apple-system, "Segoe UI", sans-serif; margin: 18px;
         background: #fafafa; color: #222; }
  h1 { font-size: 20px; margin: 0 0 2px 0; }
  .sub { color: #777; font-size: 13px; margin-bottom: 12px; }
  .controls { display: flex; gap: 18px; align-items: center; flex-wrap: wrap;
              background: #fff; border: 1px solid #ddd; border-radius: 8px;
              padding: 10px 14px; margin-bottom: 14px; }
  .controls label { font-size: 13px; }
  .controls input[type=number] { width: 64px; }
  .result { font-size: 16px; font-weight: 600; }
  .ok { color: #2a7d2a; } .bad { color: #c33; }
  .badge { font-size: 11px; color: #666; margin-left: auto; }
  .row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
  .panel { background: #fff; border: 1px solid #ddd; border-radius: 8px;
           padding: 6px; text-align: center; }
  .panel .cap { font-size: 11.5px; color: #444; margin-top: 2px; }
  canvas { display: block; }
  #wave-panel { width: 100%; }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<div class="sub">Every panel recomputes a real forward pass of the trained
model, in your browser, in the multiplicative character basis.</div>

<div class="controls">
  <label>a <input id="in-a" type="number" min="1"> </label>
  <input id="sl-a" type="range" min="1" style="width:140px">
  <label>b <input id="in-b" type="number" min="1"> </label>
  <input id="sl-b" type="range" min="1" style="width:140px">
  <button id="step">b ← g·b</button>
  <button id="anim">▶ animate</button>
  <span class="result" id="result"></span>
  <span class="badge" id="badge">checking forward pass…</span>
</div>

<div class="row" id="clock-row"></div>
<div class="row" id="cluster-row"></div>
<div class="row"><div class="panel" id="wave-panel">
  <canvas id="waves"></canvas>
  <div class="cap">per-character logit waves over candidates c (dlog order)
  — each is coset-ambiguous; the sum (black) spikes at c = a·b (green line).
  Gray: actual logits.</div>
</div></div>

<script>
// PAYLOAD-BEGIN
const P = __PAYLOAD__;
// PAYLOAD-END
// CORE-BEGIN
function runForward(P, a, b) {
  const {d_model, d_mlp, num_heads, d_head, n_ctx} = P.cfg;
  const toks = [a, b, P.eq];
  // embed + pos
  const x = [];
  for (let t = 0; t < 3; t++) {
    const v = new Float64Array(d_model);
    for (let d = 0; d < d_model; d++) v[d] = P.W_E[d][toks[t]] + P.W_pos[t][d];
    x.push(v);
  }
  // attention
  const attn = [];           // [head][key] at query t=2
  const zflat = new Float64Array(num_heads * d_head);
  for (let h = 0; h < num_heads; h++) {
    const q = new Float64Array(d_head), k = [], v = [];
    for (let t = 0; t < 3; t++) {
      const kt = new Float64Array(d_head), vt = new Float64Array(d_head);
      for (let i = 0; i < d_head; i++) {
        let sk = 0, sv = 0;
        for (let d = 0; d < d_model; d++) {
          sk += P.W_K[h][i][d] * x[t][d];
          sv += P.W_V[h][i][d] * x[t][d];
        }
        kt[i] = sk; vt[i] = sv;
      }
      k.push(kt); v.push(vt);
    }
    for (let i = 0; i < d_head; i++) {
      let s = 0;
      for (let d = 0; d < d_model; d++) s += P.W_Q[h][i][d] * x[2][d];
      q[i] = s;
    }
    const scores = [0, 1, 2].map(s => {
      let dot = 0;
      for (let i = 0; i < d_head; i++) dot += q[i] * k[s][i];
      return dot / Math.sqrt(d_head);
    });
    const mx = Math.max(...scores);
    const ex = scores.map(s => Math.exp(s - mx));
    const Z = ex[0] + ex[1] + ex[2];
    const pr = ex.map(e => e / Z);
    attn.push(pr);
    for (let i = 0; i < d_head; i++) {
      let s = 0;
      for (let t = 0; t < 3; t++) s += pr[t] * v[t][i];
      zflat[h * d_head + i] = s;
    }
  }
  const residMid = new Float64Array(d_model);
  for (let d = 0; d < d_model; d++) {
    let s = 0;
    for (let f = 0; f < zflat.length; f++) s += P.W_O[d][f] * zflat[f];
    residMid[d] = x[2][d] + s;
  }
  // MLP
  const mlpPost = new Float64Array(d_mlp);
  for (let m = 0; m < d_mlp; m++) {
    let s = P.b_in[m];
    for (let d = 0; d < d_model; d++) s += P.W_in[m][d] * residMid[d];
    mlpPost[m] = Math.max(0, s);
  }
  const residPost = new Float64Array(d_model);
  for (let d = 0; d < d_model; d++) {
    let s = P.b_out[d];
    for (let m = 0; m < d_mlp; m++) s += P.W_out[d][m] * mlpPost[m];
    residPost[d] = residMid[d] + s;
  }
  // unembed (value tokens only)
  const logits = new Float64Array(P.p);
  for (let c = 0; c < P.p; c++) {
    let s = 0;
    for (let d = 0; d < d_model; d++) s += residPost[d] * P.W_U[d][c];
    logits[c] = s;
  }
  return {logits, attn, mlpPost};
}
// CORE-END

// ----------------------------- UI ---------------------------------------
const p = P.p, n = P.p - 1;
const COL = {a: "#1f77b4", b: "#ff7f0e", ans: "#2ca02c"};
let A = 7 % p || 1, B = 12 % p || 2, timer = null;

// token order by dlog, for the wave panel x-axis
const byDlog = new Array(n);
for (let a = 1; a < p; a++) byDlog[P.dlog[a]] = a;

// precompute clock coordinates per character
const coords = {};
for (const k of P.K) {
  const {e1, e2} = P.planes[k];
  const pts = [];
  for (let a = 1; a < p; a++) {
    let u = 0, v = 0;
    for (let d = 0; d < e1.length; d++) {
      u += e1[d] * P.W_E[d][a]; v += e2[d] * P.W_E[d][a];
    }
    pts.push([u, v]);
  }
  coords[k] = pts;
}

function makePanel(row, id, w, h, cap) {
  const div = document.createElement("div");
  div.className = "panel";
  const cv = document.createElement("canvas");
  cv.id = id; cv.width = w * devicePixelRatio; cv.height = h * devicePixelRatio;
  cv.style.width = w + "px"; cv.style.height = h + "px";
  const capEl = document.createElement("div");
  capEl.className = "cap"; capEl.textContent = cap;
  div.appendChild(cv); div.appendChild(capEl);
  document.getElementById(row).appendChild(div);
  return cv;
}

const clockCv = {}, clusterCv = {};
for (const k of P.K) {
  const role = P.helpers[k] !== undefined ? `helper of χ_${P.helpers[k]}` : "primary";
  clockCv[k] = makePanel("clock-row", `clock-${k}`, 190, 190,
                         `χ_${k} clock (${role})`);
}
const attnCv = makePanel("clock-row", "attn", 190, 190, "attention from '='");
for (const k of P.K) {
  const nn = P.clusters[k].idx.length;
  clusterCv[k] = makePanel("cluster-row", `cl-${k}`, 190, 190,
    nn ? `χ_${k} cluster — ${nn} neurons, red ±θ(a)+θ(b)`
       : `χ_${k}: no neurons (free rider)`);
}
const waveCv = document.getElementById("waves");
function sizeWaves() {
  const w = document.getElementById("wave-panel").clientWidth - 16;
  waveCv.width = w * devicePixelRatio; waveCv.height = 270 * devicePixelRatio;
  waveCv.style.width = w + "px"; waveCv.style.height = "270px";
}

function ctx2d(cv) {
  const c = cv.getContext("2d");
  c.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  return c;
}

function drawClock(k) {
  const cv = clockCv[k], c = ctx2d(cv);
  const W = cv.width / devicePixelRatio, H = cv.height / devicePixelRatio;
  c.clearRect(0, 0, W, H);
  const pts = coords[k];
  let R = 0;
  for (const [u, v] of pts) R = Math.max(R, Math.hypot(u, v));
  const sc = (Math.min(W, H) / 2 - 12) / (R || 1), cx = W / 2, cy = H / 2;
  c.fillStyle = "#ccc";
  for (const [u, v] of pts) {
    c.beginPath(); c.arc(cx + u * sc, cy - v * sc, 1.6, 0, 7); c.fill();
  }
  const hand = (tok, col) => {
    const [u, v] = pts[tok - 1];
    c.strokeStyle = col; c.lineWidth = 2.2;
    c.beginPath(); c.moveTo(cx, cy); c.lineTo(cx + u * sc, cy - v * sc); c.stroke();
    c.fillStyle = col;
    c.beginPath(); c.arc(cx + u * sc, cy - v * sc, 3.4, 0, 7); c.fill();
  };
  hand(A, COL.a); hand(B, COL.b); hand((A * B) % p, COL.ans);
}

function drawAttn(attn) {
  const c = ctx2d(attnCv);
  const W = 190, H = 190, nh = attn.length;
  c.clearRect(0, 0, W, H);
  const cw = (W - 50) / 3, ch = (H - 30) / nh;
  for (let h = 0; h < nh; h++) {
    for (let s = 0; s < 3; s++) {
      const val = attn[h][s];
      c.fillStyle = `rgba(31,119,180,${val})`;
      c.fillRect(40 + s * cw, 8 + h * ch, cw - 2, ch - 2);
      c.fillStyle = val > 0.55 ? "#fff" : "#333";
      c.font = "10px sans-serif"; c.textAlign = "center";
      c.fillText(val.toFixed(2), 40 + s * cw + cw / 2, 8 + h * ch + ch / 2 + 3);
    }
    c.fillStyle = "#333"; c.textAlign = "left";
    c.fillText(`h${h}`, 18, 8 + h * ch + ch / 2 + 3);
  }
  c.textAlign = "center";
  ["a", "b", "="].forEach((t, s) =>
    c.fillText(t, 40 + s * cw + cw / 2, H - 8));
}

function drawCluster(k, mlpPost) {
  const cv = clusterCv[k], c = ctx2d(cv);
  const W = 190, H = 190, cx = W / 2, cy = H / 2, R = 78;
  c.clearRect(0, 0, W, H);
  c.strokeStyle = "#eee";
  c.beginPath(); c.arc(cx, cy, R, 0, 7); c.stroke();
  const {idx, phi} = P.clusters[k];
  if (!idx.length) {
    c.fillStyle = "#999"; c.font = "11px sans-serif"; c.textAlign = "center";
    c.fillText("rides χ_" + P.helpers[k] + "'s cluster", cx, cy);
    return;
  }
  let amax = 1e-9;
  for (const i of idx) amax = Math.max(amax, mlpPost[i]);
  c.fillStyle = "rgba(31,119,180,0.75)";
  for (let j = 0; j < idx.length; j++) {
    const r = (mlpPost[idx[j]] / amax) * R;
    const w = 0.32;
    c.beginPath(); c.moveTo(cx, cy);
    c.arc(cx, cy, r, -phi[j] - w / 2, -phi[j] + w / 2);
    c.closePath(); c.fill();
  }
  const tgt = 2 * Math.PI * k * ((P.dlog[A] + P.dlog[B]) % n) / n;
  for (const [sgn, dash] of [[1, []], [-1, [4, 3]]]) {
    c.strokeStyle = "#c33"; c.lineWidth = 1.5; c.setLineDash(dash);
    c.beginPath(); c.moveTo(cx, cy);
    c.lineTo(cx + R * Math.cos(sgn * tgt), cy - R * Math.sin(sgn * tgt));
    c.stroke();
  }
  c.setLineDash([]);
}

function drawWaves(logits) {
  const c = ctx2d(waveCv);
  const W = waveCv.width / devicePixelRatio, H = 270;
  c.clearRect(0, 0, W, H);
  const padL = 8, padR = 8, plotW = W - padL - padR;
  const xs = i => padL + plotW * i / (n - 1);
  const Lmean = logits.reduce((s, v) => s + v, 0) / p;
  // per-character projections of the live logits
  const waves = [], total = new Float64Array(n);
  for (const k of P.K) {
    const cr = P.basis[k].cos, sr = P.basis[k].sin;
    let pc = 0, ps = 0;
    for (let t = 0; t < p; t++) { pc += cr[t] * logits[t]; ps += sr[t] * logits[t]; }
    const w = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const t = byDlog[i];
      w[i] = cr[t] * pc + sr[t] * ps;
      total[i] += w[i];
    }
    waves.push(w);
  }
  let span = 1e-9;
  for (let i = 0; i < n; i++)
    span = Math.max(span, Math.abs(logits[byDlog[i]] - Lmean));
  const lane = H / (P.K.length + 1.6);
  const colors = ["#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22"];
  const drawLine = (arr, y0, amp, col, lw) => {
    c.strokeStyle = col; c.lineWidth = lw; c.beginPath();
    for (let i = 0; i < n; i++) {
      const y = y0 - arr[i] * amp;
      i ? c.lineTo(xs(i), y) : c.moveTo(xs(i), y);
    }
    c.stroke();
  };
  waves.forEach((w, j) => {
    let wmax = 1e-9;
    for (let i = 0; i < n; i++) wmax = Math.max(wmax, Math.abs(w[i]));
    drawLine(w, lane * (j + 0.7), lane * 0.42 / wmax, colors[j % 5], 1);
    c.fillStyle = colors[j % 5]; c.font = "11px sans-serif"; c.textAlign = "left";
    c.fillText("χ_" + P.K[j], padL + 2, lane * (j + 0.7) - lane * 0.32);
  });
  const y0 = lane * (P.K.length + 1.1), amp = lane * 0.95 / span;
  const tot = new Float64Array(n), act = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    tot[i] = total[i]; act[i] = logits[byDlog[i]] - Lmean;
  }
  drawLine(act, y0, amp, "#bbb", 1);
  drawLine(tot, y0, amp, "#000", 1.4);
  const xAns = xs((P.dlog[A] + P.dlog[B]) % n);
  c.strokeStyle = COL.ans; c.lineWidth = 1.4; c.setLineDash([5, 4]);
  c.beginPath(); c.moveTo(xAns, 6); c.lineTo(xAns, H - 4); c.stroke();
  c.setLineDash([]);
}

function refresh() {
  const {logits, attn, mlpPost} = runForward(P, A, B);
  for (const k of P.K) { drawClock(k); drawCluster(k, mlpPost); }
  drawAttn(attn);
  drawWaves(logits);
  let arg = 0;
  for (let cdt = 1; cdt < p; cdt++) if (logits[cdt] > logits[arg]) arg = cdt;
  const ans = (A * B) % p, ok = arg === ans;
  const el = document.getElementById("result");
  el.textContent = `${A} × ${B} ≡ ${ans} (mod ${p}) — model says ${arg} ${ok ? "✓" : "✗"}`;
  el.className = "result " + (ok ? "ok" : "bad");
  document.getElementById("in-a").value = A;
  document.getElementById("in-b").value = B;
  document.getElementById("sl-a").value = A;
  document.getElementById("sl-b").value = B;
}

for (const [id, set] of [["in-a", v => A = v], ["sl-a", v => A = v],
                         ["in-b", v => B = v], ["sl-b", v => B = v]]) {
  const el = document.getElementById(id);
  el.max = p - 1;
  el.addEventListener("input", e => {
    const v = parseInt(e.target.value, 10);
    if (v >= 1 && v < p) { set(v); refresh(); }
  });
}
document.getElementById("step").addEventListener("click",
  () => { B = (B * P.g) % p; refresh(); });
document.getElementById("anim").addEventListener("click", e => {
  if (timer) { clearInterval(timer); timer = null; e.target.textContent = "▶ animate"; }
  else {
    timer = setInterval(() => { B = (B * P.g) % p; refresh(); }, 280);
    e.target.textContent = "⏸ stop";
  }
});
window.addEventListener("resize", () => { sizeWaves(); refresh(); });

// forward-pass verification against PyTorch reference logits
(function verify() {
  let worst = 0;
  for (const chk of P.checks) {
    const {logits} = runForward(P, chk.a, chk.b);
    for (let i = 0; i < p; i++)
      worst = Math.max(worst, Math.abs(logits[i] - chk.logits[i]));
  }
  const el = document.getElementById("badge");
  el.textContent = worst < 1e-3
    ? `forward pass verified vs PyTorch (max Δ = ${worst.toExponential(1)})`
    : `FORWARD PASS MISMATCH (max Δ = ${worst.toExponential(1)})`;
  el.style.color = worst < 1e-3 ? "#2a7d2a" : "#c33";
})();

sizeWaves();
refresh();
</script>
</body>
</html>
"""


def export(run_dir: str, out_file: str | None = None) -> Path:
    S = Session.from_run(run_dir)
    payload = build_payload(S)
    run = Path(run_dir)
    title = f"mod-{payload['p']} multiplication — {run.name}"
    html = (HTML_TEMPLATE
            .replace("__TITLE__", title)
            .replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":"))))
    if out_file is None:
        exp = run.parent.parent.name
        out_file = Path("outputs/inference_viewer") / exp / f"{run.name}.html"
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-file", default=None)
    args = ap.parse_args()
    out = export(args.run_dir, args.out_file)
    print(f"Wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
