"""Lock down the ReLU harmonic-helper mechanism, two independent checks.

(A) ReLU Fourier coefficients. Confirm the doubled-frequency (2 theta)
    coefficient of ReLU(cos theta) is 2/(3 pi) ~ 0.212 for a SINGLE relu, and
    4/(3 pi) ~ 0.424 for |cos| = relu(z)+relu(-z). Note 06 originally claimed
    4/(3 pi) for a single relu, which is wrong; this pins the correct value.

(B) Enabling claim. The empirical finding is that ablating chi_{2k} from W_E
    destroys the frequency-k PRODUCT at the output. The mechanistic claim is
    that a finite-width ReLU MLP can approximate the bilinear product
    cos(k(phi_a + phi_b)) MORE accurately when given 2k input features in
    addition to k features. We test this directly: fit a width-W ReLU MLP to
    the product target on the discrete group grid, with feature sets
      primary-only : {cos k phi, sin k phi}  (a and b)
      primary+helper: + {cos 2k phi, sin 2k phi}  (a and b)
    and compare residual MSE across widths. If the helper reduces error at
    small W and the gap closes at large W, that is the capacity-pressure
    helper story, confirmed independent of the trained model.

Usage:
    python -m crypto_interp.analysis.verify_helper_mechanism [--out-dir DIR]
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# ----------------------------------------------------------------------------
# (A) ReLU / |cos| Fourier coefficients via dense FFT.
# ----------------------------------------------------------------------------
def relu_fourier_check():
    N = 200_000
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    relu = np.maximum(np.cos(theta), 0.0)
    absc = np.abs(np.cos(theta))

    def coeff(sig, m):
        # real Fourier coefficient a_m = (2/N) sum sig cos(m theta), a_0 = mean
        if m == 0:
            return sig.mean()
        return 2.0 * np.mean(sig * np.cos(m * theta))

    print("=" * 70)
    print("(A) Fourier coefficients")
    print("-" * 70)
    print(f"  ReLU(cos): a0  = {coeff(relu,0):.5f}   (1/pi      = {1/math.pi:.5f})")
    print(f"  ReLU(cos): a1  = {coeff(relu,1):.5f}   (1/2       = {0.5:.5f})")
    print(f"  ReLU(cos): a2  = {coeff(relu,2):.5f}   (2/(3pi)   = {2/(3*math.pi):.5f})")
    print(f"  ReLU(cos): a3  = {coeff(relu,3):.5f}   (0         = 0.00000)")
    print(f"  ReLU(cos): a4  = {coeff(relu,4):.5f}   (-2/(15pi) = {-2/(15*math.pi):.5f})")
    print()
    print(f"  |cos|    : a2  = {coeff(absc,2):.5f}   (4/(3pi)   = {4/(3*math.pi):.5f})")
    print()


# ----------------------------------------------------------------------------
# (B) Controlled bilinear approximation with vs without the 2k helper feature.
# ----------------------------------------------------------------------------
def features(j, n, ks):
    """Stack [cos(k phi), sin(k phi)] for each k in ks, phi = 2 pi j / n."""
    phi = 2 * math.pi * j / n
    cols = []
    for k in ks:
        cols.append(torch.cos(k * phi))
        cols.append(torch.sin(k * phi))
    return torch.stack(cols, dim=-1)


def fit_mlp(X, y, width, steps=4000, lr=5e-2, seed=0):
    torch.manual_seed(seed)
    d_in = X.shape[-1]
    net = torch.nn.Sequential(
        torch.nn.Linear(d_in, width),
        torch.nn.ReLU(),
        torch.nn.Linear(width, 1),
    ).double()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    y = y.reshape(-1, 1)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((net(X) - y) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        final = ((net(X) - y) ** 2).mean().item()
    # normalize by target variance -> fraction of variance UNexplained
    return final / y.var().item()


def bilinear_check(n=112, primary=10, widths=(1, 2, 3, 4, 6, 8), n_seeds=4, out_dir="."):
    # full grid over the cyclic group Z_n (discrete-log indices)
    ja, jb = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    ja = ja.reshape(-1).double()
    jb = jb.reshape(-1).double()
    phi_sum = 2 * math.pi * primary * (ja + jb) / n
    target = torch.cos(phi_sum)  # the clean character product at frequency k

    helper = 2 * primary
    Xa_p = features(ja, n, [primary])
    Xb_p = features(jb, n, [primary])
    X_prim = torch.cat([Xa_p, Xb_p], dim=-1)                       # 4 features

    Xa_h = features(ja, n, [primary, helper])
    Xb_h = features(jb, n, [primary, helper])
    X_help = torch.cat([Xa_h, Xb_h], dim=-1)                       # 8 features

    print("=" * 70)
    print(f"(B) Bilinear approx of cos({primary}(phi_a+phi_b)) on Z_{n} grid")
    print(f"    primary k={primary} (order {n//math.gcd(primary,n)}), "
          f"helper 2k={helper} (order {n//math.gcd(helper,n)})")
    print("-" * 70)
    print(f"  {'width':>5} | {'primary-only FVU':>18} | {'primary+helper FVU':>20} | {'ratio':>7}")
    ep_list, eh_list = [], []
    for w in widths:
        e_p = np.median([fit_mlp(X_prim, target, w, seed=s) for s in range(n_seeds)])
        e_h = np.median([fit_mlp(X_help, target, w, seed=s) for s in range(n_seeds)])
        ep_list.append(e_p); eh_list.append(e_h)
        print(f"  {w:>5} | {e_p:>18.4e} | {e_h:>20.4e} | {e_p/max(e_h,1e-12):>7.2f}")
    print()

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(widths, ep_list, "o-", color="#d62728", lw=1.8,
            label=fr"primary only $\{{\chi_{{{primary}}}\}}$")
    ax.plot(widths, eh_list, "s-", color="#1f77b4", lw=1.8,
            label=fr"primary + helper $\{{\chi_{{{primary}}}, \chi_{{{helper}}}\}}$")
    ax.set_yscale("log")
    ax.set_xlabel("MLP width (# ReLU neurons)")
    ax.set_ylabel("fraction of variance unexplained (FVU)")
    ax.set_title(fr"Approximating $\cos({primary}(\varphi_a+\varphi_b))$ with a ReLU MLP")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    out = Path(out_dir) / "helper_mechanism_fvu.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}\n")
    return


def sylow_contrast(n=112, n_seeds=4, width=2):
    """Does the helper help MORE for a Sylow-2 primary (order halves under
    doubling) than for a purely Sylow-7 primary (doubling stays same order)?
    This is the discrete-group question the continuous trig identity is blind
    to. Reported honestly either way."""
    print("=" * 70)
    print(f"(C) Sylow contrast at width={width}: error reduction from helper")
    print("-" * 70)
    cases = [
        ("Sylow-2 descent", 10),   # 10->20, order 56 -> 28 (proper descent)
        ("Sylow-2 descent", 3),    # 3 ->6,  order 112-> 56 (proper descent)
        ("Sylow-7 only",    16),   # 16->32, order 7  -> 7  (no descent)
    ]
    ja, jb = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    ja = ja.reshape(-1).double(); jb = jb.reshape(-1).double()
    print(f"  {'case':>16} | {'k':>3} | {'ord(k)':>6} | {'ord(2k)':>7} | "
          f"{'FVU prim':>10} | {'FVU help':>10} | {'ratio':>6}")
    for label, k in cases:
        tgt = torch.cos(2 * math.pi * k * (ja + jb) / n)
        Xp = torch.cat([features(ja, n, [k]),      features(jb, n, [k])], -1)
        Xh = torch.cat([features(ja, n, [k, 2*k]), features(jb, n, [k, 2*k])], -1)
        e_p = np.median([fit_mlp(Xp, tgt, width, seed=s) for s in range(n_seeds)])
        e_h = np.median([fit_mlp(Xh, tgt, width, seed=s) for s in range(n_seeds)])
        ok = n // math.gcd(k, n); o2 = n // math.gcd(2 * k, n)
        print(f"  {label:>16} | {k:>3} | {ok:>6} | {o2:>7} | "
              f"{e_p:>10.3e} | {e_h:>10.3e} | {e_p/max(e_h,1e-12):>6.2f}")
    print()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=".", help="Where to write the FVU figure.")
    args = ap.parse_args()
    relu_fourier_check()
    bilinear_check(out_dir=args.out_dir)
    sylow_contrast()
