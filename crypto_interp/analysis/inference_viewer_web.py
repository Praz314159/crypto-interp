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
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --paper: #faf9f6; --card: #fffefb; --hair: #e9e5dc; --ink: #1b1a17;
    --body: #4a463f; --muted: #807a70; --accent: #4338ca;
    --col-a: #2563eb; --col-b: #ea580c; --col-ans: #15803d;
    --col-warn: #dc2626;
    --serif: "Iowan Old Style", "Palatino", Georgia, "Times New Roman", serif;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --mono: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, monospace;
  }
  * { box-sizing: border-box; }
  body { font-family: var(--sans); margin: 0; background: var(--paper);
         color: var(--ink); -webkit-font-smoothing: antialiased; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 52px 30px 90px; }

  .eyebrow { font: 600 11.5px var(--sans); letter-spacing: 0.16em;
             text-transform: uppercase; color: var(--accent);
             margin-bottom: 16px; }
  header h1 { font: 600 42px/1.12 var(--serif); letter-spacing: -0.4px;
              margin: 0 0 16px; max-width: 22ch; }
  .lede { font: 400 17.5px/1.65 var(--serif); color: var(--body);
          max-width: 64ch; margin: 0 0 22px; }
  .lede b { font-weight: 600; color: var(--ink); }
  .tok { font-family: var(--mono); font-weight: 700; font-size: 0.92em; }
  .tok.a { color: var(--col-a); } .tok.b { color: var(--col-b); }
  .tok.ans { color: var(--col-ans); }
  .meta { display: flex; flex-wrap: wrap; gap: 8px 26px; align-items: center;
          font: 500 12.5px var(--sans); color: var(--muted);
          padding: 16px 0; border-top: 1px solid var(--hair);
          border-bottom: 1px solid var(--hair); }
  .meta b { color: var(--ink); font-weight: 650; font-family: var(--mono); }
  .chip { display: inline-flex; align-items: center; gap: 6px;
          border: 1px solid var(--hair); background: var(--card);
          border-radius: 999px; padding: 2.5px 11px;
          font: 600 12px var(--mono); }
  .chip .dot { width: 8px; height: 8px; border-radius: 50%; }
  .chip.helper { color: var(--muted); font-weight: 500; }

  .controls { position: sticky; top: 12px; z-index: 10; display: flex;
              gap: 24px; align-items: center; flex-wrap: wrap;
              background: rgba(255, 254, 251, 0.92);
              backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
              border: 1px solid var(--hair); border-radius: 16px;
              padding: 15px 22px; margin: 30px 0 44px;
              box-shadow: 0 12px 32px rgba(45, 38, 20, 0.09); }
  .knob { display: flex; align-items: center; gap: 9px; font-size: 14px; }
  .knob .sym { font: 700 17px var(--mono); }
  .knob .sym.a { color: var(--col-a); } .knob .sym.b { color: var(--col-b); }
  input[type=number] { width: 66px; font: 600 14px var(--mono);
                       padding: 4px 7px; border: 1px solid var(--hair);
                       border-radius: 8px; background: #fff; color: var(--ink); }
  input[type=range] { width: 132px; accent-color: var(--ink); }
  button { font: 600 13.5px var(--sans); padding: 8px 16px;
           border-radius: 10px; border: 1px solid var(--hair);
           background: #fff; color: var(--ink); cursor: pointer;
           transition: transform .06s ease, box-shadow .12s ease; }
  button:hover { box-shadow: 0 2px 8px rgba(45,38,20,.12); }
  button:active { transform: translateY(1px); }
  button.primary { background: var(--ink); border-color: var(--ink);
                   color: #fffefb; }
  .result { font: 650 15.5px var(--mono); letter-spacing: -0.2px; }
  .result.ok { color: var(--col-ans); } .result.bad { color: var(--col-warn); }
  .badge { font: 500 11.5px var(--sans); color: var(--muted);
           margin-left: auto; text-align: right; }

  .stage { background: var(--card); border: 1px solid var(--hair);
           border-radius: 18px; padding: 30px 34px 26px; margin-bottom: 30px;
           box-shadow: 0 1px 3px rgba(45, 38, 20, 0.04); }
  .stage h2 { font: 600 22.5px var(--serif); margin: 0 0 10px;
              display: flex; align-items: center; gap: 14px;
              letter-spacing: -0.2px; }
  .stage h2 .num { width: 30px; height: 30px; border-radius: 50%;
                   border: 1.5px solid var(--ink); color: var(--ink);
                   font: 600 15px var(--serif); display: inline-flex;
                   align-items: center; justify-content: center; flex: none; }
  .stage p.exp { color: var(--body); font-size: 14px; line-height: 1.72;
                 margin: 6px 0 22px; max-width: 80ch; }
  .stage p.exp b { color: var(--ink); font-weight: 600; }
  .clock-grid { display: flex; gap: 22px; flex-wrap: wrap; }
  .cell { text-align: center; }
  .cell .lab { font: 500 12px var(--mono); color: var(--muted);
               margin-top: 7px; }
  .cell .lab b { color: var(--ink); font-weight: 700; }
  canvas { display: block; }
  .dyn { font: 500 13px var(--mono); color: var(--muted); margin-top: 16px;
         padding-top: 12px; border-top: 1px dashed var(--hair); }
  .dyn b { color: var(--ink); }
  #coset-box, #wave-box { width: 100%; }
  .legend { font-size: 12.5px; color: var(--muted); }
  .swatch { display: inline-block; width: 10px; height: 10px;
            border-radius: 3px; margin: 0 5px 0 12px; vertical-align: -1px; }
  #tip { position: fixed; pointer-events: none; z-index: 50; display: none;
         background: var(--ink); color: #fffefb; border-radius: 8px;
         padding: 5px 10px; font: 600 12px var(--mono);
         box-shadow: 0 6px 18px rgba(0,0,0,.25); }
  footer { margin-top: 44px; color: var(--muted); font-size: 12.5px;
           line-height: 1.7; border-top: 1px solid var(--hair);
           padding-top: 16px; }
</style>
</head>
<body>
<div class="wrap">

<header>
  <div class="eyebrow">Mechanistic interpretability · live model dissection</div>
  <h1>Watch a transformer multiply</h1>
  <p class="lede">A 1-layer transformer trained on
  <b>a × b mod __P__</b> doesn't memorize — it invents an algorithm:
  represent every number as angles on a handful of clocks, <b>rotate</b> to
  multiply, and let the clocks vote on the answer. The trained model is
  embedded in this page and runs <b>live in your browser</b>; every figure
  below is a real forward pass, recomputed as you change
  <span class="tok a">a</span> and <span class="tok b">b</span>.</p>
  <div class="meta" id="meta"></div>
</header>

<div class="controls">
  <div class="knob"><span class="sym a">a</span>
    <input id="in-a" type="number" min="1">
    <input id="sl-a" type="range" min="1"></div>
  <div class="knob"><span class="sym b">b</span>
    <input id="in-b" type="number" min="1">
    <input id="sl-b" type="range" min="1"></div>
  <button class="primary" id="play">▶ replay walkthrough</button>
  <button id="step" title="multiply b by the primitive root g — the algorithm's unit step">b ← g·b</button>
  <span class="result" id="result"></span>
  <span class="badge" id="badge">checking forward pass…</span>
</div>

<section class="stage">
  <h2><span class="num">1</span>Look up the angles</h2>
  <p class="exp">The embedding stores every token as a position on
  <b>__NK__ clocks</b> — one per character χ<sub>k</sub> the model learned to
  use. Token x sits at angle θ<sub>k</sub>(x) = 2πk·dlog(x)/(p−1), where dlog
  is the discrete logarithm base g=__G__. Each circle below is the actual
  embedding matrix W<sub>E</sub> projected onto one character's plane; gray
  dots are all p−1 tokens. The model "reads" <b style="color:var(--col-a)">a</b>
  and <b style="color:var(--col-b)">b</b> by looking up their hands.</p>
  <div class="clock-grid" id="row-lookup"></div>
  <div class="dyn" id="dyn-lookup"></div>
</section>

<section class="stage">
  <h2><span class="num">2</span>Add the angles — this is the multiplication</h2>
  <p class="exp">Multiplying numbers is <b>adding their discrete logs</b>, and
  adding logs is <b>rotating a clock hand</b>. Watch the
  <b style="color:var(--col-ans)">green hand</b> start at a's angle and rotate
  by b's angle: it lands exactly on the gray dot where the embedding stores
  a·b — on every clock at once. A <b>helper</b> character χ<sub>2m</sub> is
  the same motion at double speed: it reuses χ<sub>m</sub>'s machinery and
  costs the model nothing extra (the dashed green hand is the primary's angle
  doubled — it coincides with the helper's own hand).</p>
  <div class="clock-grid" id="row-add"></div>
</section>

<section class="stage">
  <h2><span class="num">3</span>Neuron clusters detect the summed angle</h2>
  <p class="exp">In the MLP, each <b>primary</b> character owns a small
  cluster of ReLU neurons, each tuned to a preferred phase — its position
  around the circle below. Bar length is that neuron's <b>actual activation
  on this forward pass</b>. The active bump always points at
  θ<sub>k</sub>(a)+θ<sub>k</sub>(b) (red line; the dashed mirror exists
  because a real cos/sin basis cannot tell k from −k). Helper characters have
  <b>zero neurons of their own</b> — the ReLU's second harmonic produces their
  signal from the primary's bump for free.</p>
  <div class="clock-grid" id="row-cluster"></div>
</section>

<section class="stage">
  <h2><span class="num">4</span>Each character votes for a coset — the CRT step</h2>
  <p class="exp">One clock cannot name the answer. Character χ<sub>k</sub>
  only measures angles at frequency k, so every candidate c with the same
  k·dlog(c) looks identical to it: exactly <b>gcd(k, p−1) candidates</b>
  survive each character's vote (colored cells below; candidates ordered by
  dlog). The algorithm works because the surviving sets of different
  characters <b>intersect in a single candidate</b> — the model's
  approximate Chinese-Remainder argument. This is why it must learn several
  characters whose frequencies jointly cover p−1, and why a seed whose K
  fails to cover (like the CRT-failure regime) cannot reach zero loss.</p>
  <div id="coset-box"><canvas id="cosets"></canvas></div>
</section>

<section class="stage">
  <h2><span class="num">5</span>Interference turns votes into logits</h2>
  <p class="exp">The unembed converts each character's vote into a cosine
  wave over all candidates: peaks on the character's surviving coset. The
  waves are individually ambiguous, but their <b>sum spikes only at
  c = a·b</b> — constructive interference exactly where every character
  agrees. <span class="legend"><span class="swatch" style="background:#999"></span>actual
  model logits <span class="swatch" style="background:#000"></span>sum of the K
  character waves <span class="swatch" style="background:var(--col-ans)"></span>true
  answer</span></p>
  <div id="wave-box"><canvas id="waves"></canvas></div>
</section>

<footer id="foot"></footer>

</div>
<div id="tip"></div>
<script>
// PAYLOAD-BEGIN
const P = __PAYLOAD__;
// PAYLOAD-END
// CORE-BEGIN
function runForward(P, a, b) {
  const {d_model, d_mlp, num_heads, d_head} = P.cfg;
  const toks = [a, b, P.eq];
  const x = [];
  for (let t = 0; t < 3; t++) {
    const v = new Float64Array(d_model);
    for (let d = 0; d < d_model; d++) v[d] = P.W_E[d][toks[t]] + P.W_pos[t][d];
    x.push(v);
  }
  const attn = [];
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
  const logits = new Float64Array(P.p);
  for (let c = 0; c < P.p; c++) {
    let s = 0;
    for (let d = 0; d < d_model; d++) s += residPost[d] * P.W_U[d][c];
    logits[c] = s;
  }
  return {logits, attn, mlpPost};
}
// CORE-END

// ----------------------------- setup -------------------------------------
const $ = id => document.getElementById(id);
const p = P.p, n = P.p - 1, DPR = window.devicePixelRatio || 1;
const COL = {a: "#2563eb", b: "#ea580c", ans: "#16a34a", warn: "#dc2626"};
const CHAR_COLORS = ["#7c3aed", "#0891b2", "#db2777", "#a16207", "#4d7c0f",
                     "#be123c", "#0e7490"];
const charColor = {};
P.K.forEach((k, i) => charColor[k] = CHAR_COLORS[i % CHAR_COLORS.length]);
const isHelper = k => P.helpers[k] !== undefined;
const TAU = 2 * Math.PI;
let A = 7 % p || 1, B = 12 % p || 2;
let fwd = null;          // current forward-pass results
let anim = null;         // walkthrough animation handle

const byDlog = new Array(n);
for (let a = 1; a < p; a++) byDlog[P.dlog[a]] = a;

// clock geometry: coordinates of all tokens in each character plane,
// plus each plane's rotational orientation (chirality)
const geom = {};
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
  const ang = tok => Math.atan2(pts[tok - 1][1], pts[tok - 1][0]);
  // orientation: stepping x -> g*x should advance angle by ±2πk/n
  let s = 0;
  for (let j = 0; j < 6; j++) {
    const x0 = byDlog[j], x1 = byDlog[j + 1];
    let d = ang(x1) - ang(x0);
    while (d > Math.PI) d -= TAU;
    while (d < -Math.PI) d += TAU;
    s += d;
  }
  geom[k] = {pts, ang, orient: Math.sign(s) || 1};
}

function gcd(a, b) { while (b) { [a, b] = [b, a % b]; } return a; }

// ----------------------------- panels -------------------------------------
function makeCanvas(parent, w, h, label) {
  const cell = document.createElement("div");
  cell.className = "cell";
  const cv = document.createElement("canvas");
  cv.width = w * DPR; cv.height = h * DPR;
  cv.style.width = w + "px"; cv.style.height = h + "px";
  cell.appendChild(cv);
  const lab = document.createElement("div");
  lab.className = "lab"; lab.innerHTML = label;
  cell.appendChild(lab);
  $(parent).appendChild(cv.parentElement === cell ? cell : cell);
  return cv;
}

const CW = 216;
const lookupCv = {}, addCv = {}, clusterCv = {};
for (const k of P.K) {
  const role = isHelper(k)
    ? `<b>χ_${k}</b> · helper of χ_${P.helpers[k]}`
    : `<b>χ_${k}</b> · primary`;
  lookupCv[k] = makeCanvas("row-lookup", CW, CW, role);
  addCv[k] = makeCanvas("row-add", CW, CW, role);
}
for (const k of P.K) {
  const nn = P.clusters[k].idx.length;
  clusterCv[k] = makeCanvas("row-cluster", CW, CW,
    nn ? `<b>χ_${k}</b> · ${nn} neurons` : `<b>χ_${k}</b> · 0 neurons (free rider)`);
}

function ctx2d(cv) {
  const c = cv.getContext("2d");
  c.setTransform(DPR, 0, 0, DPR, 0, 0);
  return c;
}

function clockBase(cv, k) {
  const c = ctx2d(cv);
  c.clearRect(0, 0, CW, CW);
  const pts = geom[k].pts;
  let R = 0;
  for (const [u, v] of pts) R = Math.max(R, Math.hypot(u, v));
  const sc = (CW / 2 - 16) / (R || 1), cx = CW / 2, cy = CW / 2;
  const ring = CW / 2 - 16;
  // face: ring + cardinal ticks + center pin
  c.strokeStyle = "#e9e5dc"; c.lineWidth = 1.2;
  c.beginPath(); c.arc(cx, cy, ring, 0, TAU); c.stroke();
  c.strokeStyle = "#dedacf";
  for (let q = 0; q < 4; q++) {
    const a = q * Math.PI / 2;
    c.beginPath();
    c.moveTo(cx + (ring - 4) * Math.cos(a), cy - (ring - 4) * Math.sin(a));
    c.lineTo(cx + (ring + 3) * Math.cos(a), cy - (ring + 3) * Math.sin(a));
    c.stroke();
  }
  c.fillStyle = "#cfccc3";
  for (const [u, v] of pts) {
    c.beginPath(); c.arc(cx + u * sc, cy - v * sc, 1.8, 0, TAU); c.fill();
  }
  c.fillStyle = "#a8a499";
  c.beginPath(); c.arc(cx, cy, 2.2, 0, TAU); c.fill();
  cv._map = {k, sc, cx, cy};   // for hover inspection
  return {c, sc, cx, cy, R: ring};
}

function hand(c, cx, cy, x, y, col, w, dash) {
  c.save();
  c.strokeStyle = col; c.lineWidth = w; c.lineCap = "round";
  if (dash) c.setLineDash(dash);
  c.beginPath(); c.moveTo(cx, cy); c.lineTo(x, y); c.stroke();
  c.restore();
  c.fillStyle = col;
  c.beginPath(); c.arc(x, y, w + 1.7, 0, TAU); c.fill();
  c.fillStyle = "#ffffff";
  c.beginPath(); c.arc(x, y, Math.max(w - 1, 0.8), 0, TAU); c.fill();
}

function tokenXY(k, tok, sc, cx, cy) {
  const [u, v] = geom[k].pts[tok - 1];
  return [cx + u * sc, cy - v * sc];
}

function drawLookup(k) {
  const {c, sc, cx, cy} = clockBase(lookupCv[k], k);
  const [ax, ay] = tokenXY(k, A, sc, cx, cy);
  const [bx, by] = tokenXY(k, B, sc, cx, cy);
  hand(c, cx, cy, ax, ay, COL.a, 2.4);
  hand(c, cx, cy, bx, by, COL.b, 2.4);
  c.font = "600 11px sans-serif";
  c.fillStyle = COL.a; c.fillText("a", ax + 5, ay - 5);
  c.fillStyle = COL.b; c.fillText("b", bx + 5, by - 5);
}

// t ∈ [0,1]: green hand sweeps from a's angle by b's angle
function drawAdd(k, t) {
  const {c, sc, cx, cy} = clockBase(addCv[k], k);
  const G = geom[k];
  const ans = (A * B) % p;
  const [ax, ay] = tokenXY(k, A, sc, cx, cy);
  hand(c, cx, cy, ax, ay, COL.a, 1.6);
  const [bx, by] = tokenXY(k, B, sc, cx, cy);
  hand(c, cx, cy, bx, by, COL.b, 1.6);

  const angA = G.ang(A), angAns = G.ang(ans);
  let delta = G.orient * (angAns - angA);
  while (delta < 0) delta += TAU;
  while (delta >= TAU) delta -= TAU;
  const phi = angA + G.orient * delta * t;
  const r = Math.hypot(...G.pts[ans - 1]) * sc;
  const gx = cx + r * Math.cos(phi), gy = cy - r * Math.sin(phi);

  // swept arc
  c.save();
  c.strokeStyle = COL.ans; c.globalAlpha = 0.35; c.lineWidth = 5;
  c.beginPath();
  if (G.orient > 0) c.arc(cx, cy, r * 0.55, -angA, -(angA + delta * t), true);
  else c.arc(cx, cy, r * 0.55, -angA, -(angA - delta * t), false);
  c.stroke();
  c.restore();

  // target dot for a·b
  const [tx, ty] = tokenXY(k, ans, sc, cx, cy);
  c.strokeStyle = COL.ans; c.lineWidth = 1.2;
  c.beginPath(); c.arc(tx, ty, 5.5, 0, TAU); c.stroke();

  hand(c, cx, cy, gx, gy, COL.ans, 2.6);

  // helper χ_{2m}: dashed hand at TWICE the primary's measured rotation —
  // θ_{2m}(x) = 2·θ_m(x), so doubling χ_m's angle must land on χ_{2m}'s own
  // hand. Coincidence of dashed and solid is the free-rider relation.
  if (isHelper(k) && t >= 1) {
    const m = P.helpers[k], Gm = geom[m];
    if (Gm) {
      let relM = Gm.orient * (Gm.ang(ans) - Gm.ang(1));
      while (relM < 0) relM += TAU;
      const phiPred = G.ang(1) + G.orient * ((2 * relM) % TAU);
      const rH = Math.hypot(...G.pts[ans - 1]) * sc * 0.8;
      hand(c, cx, cy, cx + rH * Math.cos(phiPred),
           cy - rH * Math.sin(phiPred), COL.ans, 1.3, [5, 4]);
    }
  }
  if (t >= 1) {
    c.font = "600 11px sans-serif"; c.fillStyle = COL.ans;
    c.fillText("a·b ✓", tx + 7, ty + 3);
  }
}

function drawCluster(k, t) {
  const cv = clusterCv[k], c = ctx2d(cv);
  c.clearRect(0, 0, CW, CW);
  const cx = CW / 2, cy = CW / 2, R = CW / 2 - 16;
  c.strokeStyle = "#eeede9";
  c.beginPath(); c.arc(cx, cy, R, 0, TAU); c.stroke();
  const {idx, phi} = P.clusters[k];
  if (!idx.length) {
    c.fillStyle = "#9c9a94"; c.font = "12px sans-serif"; c.textAlign = "center";
    c.fillText("no neurons —", cx, cy - 8);
    c.fillText(`rides χ_${P.helpers[k]}'s cluster`, cx, cy + 10);
    c.textAlign = "left";
    return;
  }
  // resting spokes: every neuron's preferred phase, even when silent
  c.strokeStyle = "#f0ede6"; c.lineWidth = 1;
  for (let j = 0; j < idx.length; j++) {
    c.beginPath(); c.moveTo(cx, cy);
    c.lineTo(cx + R * Math.cos(-phi[j]), cy + R * Math.sin(-phi[j]));
    c.stroke();
  }
  let amax = 1e-9;
  for (const i of idx) amax = Math.max(amax, fwd.mlpPost[i]);
  c.fillStyle = charColor[k] + "c0";
  for (let j = 0; j < idx.length; j++) {
    const r = t * (fwd.mlpPost[idx[j]] / amax) * R;
    const w = 0.34;
    c.beginPath(); c.moveTo(cx, cy);
    c.arc(cx, cy, r, -phi[j] - w / 2, -phi[j] + w / 2);
    c.closePath(); c.fill();
  }
  const tgt = TAU * k * ((P.dlog[A] + P.dlog[B]) % n) / n;
  for (const [sgn, dash] of [[1, []], [-1, [4, 3]]]) {
    c.strokeStyle = COL.warn; c.lineWidth = 1.5; c.setLineDash(dash);
    c.beginPath(); c.moveTo(cx, cy);
    c.lineTo(cx + R * Math.cos(sgn * tgt), cy - R * Math.sin(sgn * tgt));
    c.stroke();
  }
  c.setLineDash([]);
}

function sizeWide(cv, h) {
  const w = cv.parentElement.clientWidth;
  cv.width = w * DPR; cv.height = h * DPR;
  cv.style.width = w + "px"; cv.style.height = h + "px";
  return w;
}

// stage 4: coset elimination strips, one per character + intersection
function drawCosets(t) {
  const cv = $("cosets");
  const W = cv.width / DPR, rowH = 30, padL = 120, padR = 16;
  const c = ctx2d(cv);
  c.clearRect(0, 0, W, cv.height / DPR);
  const cellW = (W - padL - padR) / n;
  const xAns = (P.dlog[A] + P.dlog[B]) % n;
  const nShow = Math.ceil(P.K.length * Math.min(1, t * P.K.length /
                          Math.max(P.K.length, 1)));
  const survives = k => {
    const out = new Array(n).fill(false);
    const G = gcd(k, n), step = n / G;
    for (let j = 0; j < G; j++) out[(xAns + j * step) % n] = true;
    return out;
  };
  let inter = new Array(n).fill(true);
  P.K.forEach((k, row) => {
    const y = 8 + row * rowH;
    const vis = (t * (P.K.length + 1)) > row;
    const sv = survives(k);
    for (let i = 0; i < n; i++) inter[i] = inter[i] && sv[i];
    if (!vis) return;
    c.font = "600 12px sans-serif"; c.fillStyle = charColor[k];
    c.fillText(`χ_${k}`, 8, y + 15);
    c.font = "11px sans-serif"; c.fillStyle = "#9c9a94";
    c.fillText(`keeps ${gcd(k, n)}`, 44, y + 15);
    for (let i = 0; i < n; i++) {
      c.fillStyle = sv[i] ? charColor[k] : "#f0ede6";
      c.beginPath();
      if (c.roundRect) c.roundRect(padL + i * cellW, y,
                                   Math.max(cellW - 1.2, 1), rowH - 9, 2);
      else c.rect(padL + i * cellW, y, Math.max(cellW - 1.2, 1), rowH - 9);
      c.fill();
    }
  });
  const yI = 8 + P.K.length * rowH + 6;
  if (t * (P.K.length + 1) > P.K.length) {
    c.font = "600 12px sans-serif"; c.fillStyle = "#1c1c1a";
    c.fillText("∩ all", 8, yI + 15);
    let count = 0;
    for (let i = 0; i < n; i++) {
      if (inter[i]) {
        count++;
        c.save();
        c.shadowColor = COL.ans; c.shadowBlur = 9;
        c.fillStyle = COL.ans;
        c.fillRect(padL + i * cellW - 1, yI - 2,
                   Math.max(cellW + 0.8, 3), rowH - 5);
        c.restore();
      } else {
        c.fillStyle = "#f0ede6";
        c.beginPath();
        if (c.roundRect) c.roundRect(padL + i * cellW, yI,
                                     Math.max(cellW - 1.2, 1), rowH - 9, 2);
        else c.rect(padL + i * cellW, yI, Math.max(cellW - 1.2, 1), rowH - 9);
        c.fill();
      }
    }
    c.font = "11px sans-serif"; c.fillStyle = "#9c9a94";
    c.fillText(`${count} left`, 44, yI + 15);
    const ans = (A * B) % p;
    c.font = "600 11.5px sans-serif"; c.fillStyle = COL.ans;
    const x = padL + xAns * cellW;
    c.fillText(`c = ${ans}`, Math.min(x + 5, W - 60), yI + rowH + 8);
  }
  void nShow;
}

function drawWaves(t) {
  const cv = $("waves");
  const W = cv.width / DPR, H = cv.height / DPR;
  const c = ctx2d(cv);
  c.clearRect(0, 0, W, H);
  const padL = 36, padR = 10, plotW = W - padL - padR;
  const xs = i => padL + plotW * i / (n - 1);
  const logits = fwd.logits;
  const Lmean = logits.reduce((s, v) => s + v, 0) / p;
  const waves = [], total = new Float64Array(n);
  for (const k of P.K) {
    const cr = P.basis[k].cos, sr = P.basis[k].sin;
    let pc = 0, ps = 0;
    for (let tk = 0; tk < p; tk++) { pc += cr[tk] * logits[tk]; ps += sr[tk] * logits[tk]; }
    const w = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const tok = byDlog[i];
      w[i] = cr[tok] * pc + sr[tok] * ps;
      total[i] += w[i];
    }
    waves.push(w);
  }
  const lane = H / (P.K.length + 2.1);
  const iMax = Math.max(2, Math.floor(n * Math.min(t, 1)));
  const line = (arr, y0, amp, col, lw) => {
    c.strokeStyle = col; c.lineWidth = lw; c.beginPath();
    for (let i = 0; i < iMax; i++) {
      const y = y0 - arr[i] * amp;
      i ? c.lineTo(xs(i), y) : c.moveTo(xs(i), y);
    }
    c.stroke();
  };
  waves.forEach((w, j) => {
    let wmax = 1e-9;
    for (let i = 0; i < n; i++) wmax = Math.max(wmax, Math.abs(w[i]));
    const y0 = lane * (j + 0.75);
    c.strokeStyle = "#f0ede6"; c.lineWidth = 1;
    c.beginPath(); c.moveTo(padL, y0); c.lineTo(W - padR, y0); c.stroke();
    line(w, y0, lane * 0.4 / wmax, charColor[P.K[j]], 1.2);
    c.fillStyle = charColor[P.K[j]];
    c.font = "700 11.5px ui-monospace, Menlo, monospace"; c.textAlign = "left";
    c.fillText("χ_" + P.K[j], 4, y0 + 4);
  });
  let span = 1e-9;
  for (let i = 0; i < n; i++)
    span = Math.max(span, Math.abs(logits[byDlog[i]] - Lmean));
  const y0 = lane * (P.K.length + 1.35), amp = lane * 1.15 / span;
  const act = new Float64Array(n);
  for (let i = 0; i < n; i++) act[i] = logits[byDlog[i]] - Lmean;
  c.strokeStyle = "#f0ede6"; c.lineWidth = 1;
  c.beginPath(); c.moveTo(padL, y0); c.lineTo(W - padR, y0); c.stroke();
  line(act, y0, amp, "#bcb8b0", 1);
  line(total, y0, amp, "#15140f", 1.6);
  c.fillStyle = "#15140f";
  c.font = "700 12px ui-monospace, Menlo, monospace";
  c.fillText("Σ", 14, y0 + 4);
  // dot on the spike once the draw has reached it
  let iPk = 0;
  for (let i = 1; i < n; i++) if (total[i] > total[iPk]) iPk = i;
  if (iPk < iMax) {
    c.fillStyle = COL.ans;
    c.beginPath(); c.arc(xs(iPk), y0 - total[iPk] * amp, 3.6, 0, TAU); c.fill();
  }
  const xAns = xs((P.dlog[A] + P.dlog[B]) % n);
  c.strokeStyle = COL.ans; c.lineWidth = 1.4; c.setLineDash([6, 4]);
  c.beginPath(); c.moveTo(xAns, 6); c.lineTo(xAns, H - 16); c.stroke();
  c.setLineDash([]);
  c.fillStyle = COL.ans; c.textAlign = "center";
  c.fillText("a·b", xAns, H - 3);
  c.textAlign = "left";
  c.fillStyle = "#9c9a94"; c.font = "11px sans-serif";
  c.fillText("candidates c, ordered by dlog(c) →", padL, H - 3);
}

// ----------------------------- orchestration ------------------------------
function renderAll(opts) {
  const o = Object.assign({add: 1, cluster: 1, coset: 1, wave: 1}, opts);
  fwd = runForward(P, A, B);
  for (const k of P.K) {
    drawLookup(k);
    drawAdd(k, o.add);
    drawCluster(k, o.cluster);
  }
  drawCosets(o.coset);
  drawWaves(o.wave);
  let arg = 0;
  for (let cdt = 1; cdt < p; cdt++) if (fwd.logits[cdt] > fwd.logits[arg]) arg = cdt;
  const ans = (A * B) % p, ok = arg === ans;
  const el = $("result");
  el.textContent = `${A} × ${B} ≡ ${ans} — model says ${arg} ${ok ? "✓" : "✗"}`;
  el.className = "result " + (ok ? "ok" : "bad");
  $("in-a").value = A; $("sl-a").value = A;
  $("in-b").value = B; $("sl-b").value = B;
  $("dyn-lookup").innerHTML =
    `dlog(<b>${A}</b>) = <b>${P.dlog[A]}</b> &nbsp;·&nbsp; ` +
    `dlog(<b>${B}</b>) = <b>${P.dlog[B]}</b> &nbsp;·&nbsp; ` +
    `dlog(a·b) = ${P.dlog[A]} + ${P.dlog[B]} ≡ <b>${(P.dlog[A] + P.dlog[B]) % n}</b> (mod ${n})`;
}

function walkthrough() {
  if (anim) cancelAnimationFrame(anim);
  const T = [1400, 900, 1400, 1100];   // add, cluster, coset, wave durations
  const start = performance.now();
  const ease = u => u < 0 ? 0 : u > 1 ? 1 : u * u * (3 - 2 * u);
  function frame(now) {
    const el = now - start;
    let acc = 0;
    const ph = [];
    for (const d of T) { ph.push((el - acc) / d); acc += d; }
    renderAll({
      add: ease(ph[0]),
      cluster: ease(ph[1]),
      coset: ease(ph[2]),
      wave: ease(ph[3]),
    });
    if (el < acc) anim = requestAnimationFrame(frame);
    else anim = null;
  }
  anim = requestAnimationFrame(frame);
}

// meta line + footer
{
  const chips = P.K.map(k => {
    const dot = `<span class="dot" style="background:${charColor[k]}"></span>`;
    return isHelper(k)
      ? `<span class="chip helper">${dot}χ_${k} = 2·χ_${P.helpers[k]}</span>`
      : `<span class="chip">${dot}χ_${k}</span>`;
  }).join(" ");
  $("meta").innerHTML =
    `<span>p = <b>${p}</b></span>` +
    `<span>primitive root g = <b>${P.g}</b></span>` +
    `<span>d_model = <b>${P.cfg.d_model}</b></span>` +
    `<span>d_mlp = <b>${P.cfg.d_mlp}</b></span>` +
    `<span>characters: ${chips}</span>`;
  $("foot").innerHTML =
    `Model: 1-layer attention + ReLU MLP transformer (no LayerNorm), trained ` +
    `to grok modular multiplication. The page is fully self-contained — the ` +
    `weights are embedded and the forward pass runs in JavaScript, verified ` +
    `against PyTorch on load. Hover any gray dot on a clock to identify its ` +
    `token. Built with the crypto_interp library.`;
}

// hover inspection on clock faces: identify the token under the cursor
{
  const tip = $("tip");
  const attach = cv => cv.addEventListener("mousemove", ev => {
    const m = cv._map;
    if (!m) return;
    const r = cv.getBoundingClientRect();
    const mx = ev.clientX - r.left, my = ev.clientY - r.top;
    let best = -1, bd = 144; // 12px snap radius
    const pts = geom[m.k].pts;
    for (let t = 1; t < p; t++) {
      const dx = m.cx + pts[t - 1][0] * m.sc - mx;
      const dy = m.cy - pts[t - 1][1] * m.sc - my;
      const d2 = dx * dx + dy * dy;
      if (d2 < bd) { bd = d2; best = t; }
    }
    if (best > 0) {
      tip.style.display = "block";
      tip.style.left = (ev.clientX + 14) + "px";
      tip.style.top = (ev.clientY - 10) + "px";
      tip.textContent = `token ${best} · dlog ${P.dlog[best]}`;
    } else tip.style.display = "none";
  });
  for (const k of P.K) { attach(lookupCv[k]); attach(addCv[k]); }
  document.addEventListener("mouseout", () => tip.style.display = "none");
}

for (const [id, set] of [["in-a", v => A = v], ["sl-a", v => A = v],
                         ["in-b", v => B = v], ["sl-b", v => B = v]]) {
  const el = $(id);
  el.max = p - 1;
  el.addEventListener("input", e => {
    const v = parseInt(e.target.value, 10);
    if (v >= 1 && v < p) { set(v); renderAll(); }
  });
}
$("step").addEventListener("click", () => { B = (B * P.g) % p; walkthrough(); });
$("play").addEventListener("click", walkthrough);
window.addEventListener("resize", () => {
  sizeWide($("cosets"), 8 + (P.K.length + 1) * 30 + 24);
  sizeWide($("waves"), 330);
  renderAll();
});

(function verify() {
  let worst = 0;
  for (const chk of P.checks) {
    const {logits} = runForward(P, chk.a, chk.b);
    for (let i = 0; i < p; i++)
      worst = Math.max(worst, Math.abs(logits[i] - chk.logits[i]));
  }
  const el = $("badge");
  el.textContent = worst < 1e-3
    ? `forward pass verified vs PyTorch (Δ ≤ ${worst.toExponential(1)})`
    : `FORWARD PASS MISMATCH (Δ = ${worst.toExponential(1)})`;
  el.style.color = worst < 1e-3 ? "#16a34a" : "#dc2626";
})();

sizeWide($("cosets"), 8 + (P.K.length + 1) * 30 + 24);
sizeWide($("waves"), 330);
renderAll();
walkthrough();
</script>
</body>
</html>
"""


def export(run_dir: str, out_file: str | None = None) -> Path:
    S = Session.from_run(run_dir)
    payload = build_payload(S)
    run = Path(run_dir)
    html = (HTML_TEMPLATE
            .replace("__TITLE__", f"mod-{payload['p']} multiplication — {run.name}")
            .replace("__P__", str(payload["p"]))
            .replace("__G__", str(payload["g"]))
            .replace("__NK__", str(len(payload["K"])))
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
