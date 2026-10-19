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
