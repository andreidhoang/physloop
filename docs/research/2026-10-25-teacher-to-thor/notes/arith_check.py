# Re-derivation of the 2026-10-25 decision doc's arithmetic (power, latency, budget). Python stdlib only.
from math import sqrt, ceil
from statistics import NormalDist
Z = NormalDist().inv_cdf

def connor_n(delta, psi, alpha=0.05, power=0.8, sided=1):
    """Connor 1987 sample size for paired binary (McNemar) given difference delta and discordance psi."""
    za = Z(1 - alpha / sided); zb = Z(power)
    return (za * sqrt(psi) + zb * sqrt(psi - delta**2))**2 / delta**2

def connor_mde(n, psi, alpha=0.05, power=0.8, sided=2):
    lo, hi = 1e-6, sqrt(psi) - 1e-9
    for _ in range(200):
        mid = (lo + hi) / 2
        if connor_n(mid, psi, alpha, power, sided) > n: lo = mid
        else: hi = mid
    return hi

print("== MDE at n=441 (Connor; plan v3 quotes 1.94/2.72/3.90 pp exact) ==")
for psi in (0.021, 0.04, 0.08):
    print(f"psi={psi:.3f}: two-sided MDE {100*connor_mde(441, psi):.2f} pp; one-sided {100*connor_mde(441, psi, sided=1):.2f} pp")

print("\n== P1 gate: at-fault collisions down >=20% relative, on 441 held-out scenes ==")
print("base p0 | delta(pp) | psi assumed | n needed (1-sided, 80%) | certifiable on 441? | on 441x3 seeds (deff 2.2 -> n_eff 601)?")
for p0 in (0.04, 0.08, 0.16, 0.19):
    d = 0.2 * p0
    for psi in (d, 2 * d + 0.01, 0.04, 0.08):
        if psi < d**2 + 1e-12 or psi < d: continue
        n = connor_n(d, psi, sided=1)
        print(f"{p0:5.0%} | {100*d:5.2f} | {psi:6.3f} | {ceil(n):6d} | {'yes' if n <= 441 else 'NO'} | {'yes' if n <= 601 else 'NO'}")

print("\n== P2 gate: student >= 90% of teacher score (as a paired NI margin on per-scene score) ==")
for s in (0.5, 0.7, 0.9):
    m = 0.1 * s
    print(f"teacher mean score {s}: margin = {100*m:.1f} pp vs MDE ~2.7 pp at psi=4% -> certifiable if pinned protocol")
print(f"card protocol drift 0.81 -> 1.37 = {1.37/0.81-1:.0%} with unchanged weights, i.e. {((1.37/0.81)-1)/0.10:.1f}x the 10% margin")

print("\n== Latency bound, doc formula t = 2*N*T/(eta*F) + n_tok*B_bb/BW + K*B_exp/BW ==")
F, BW, eta = 517e12, 273e9, 0.4
def row(name, N_bb, bpp_pre, bpp_dec, n_tok, N_exp, bpp_exp, K, T):
    pre = 2 * N_bb * T / (eta * F)
    dec = n_tok * N_bb * bpp_dec / BW
    exp = K * N_exp * bpp_exp / BW
    return name, pre * 1e3, dec * 1e3, exp * 1e3, (pre + dec + exp) * 1e3
rows = []
for T in (2560, 2880):
    rows += [
        row(f"10B FP8, K=10, T={T}", 8.2e9, 1, 1, 20, 2.3e9, 1, 10, T),
        row(f"10B NVFP4 dec, FP8 exp, K=10, T={T}", 8.2e9, 1, 0.5625, 20, 2.3e9, 1, 10, T),
        row(f"2B NVFP4, 0.5B exp FP8, K=4, T={T}", 2e9, 1, 0.5625, 20, 0.5e9, 1, 4, T),
        row(f"2B no CoC, K=2, T={T}", 2e9, 1, 0.5625, 0, 0.5e9, 1, 2, T),
    ]
print(f"{'config':42s} prefill  decode  expert   total (ms)")
for r in rows: print(f"{r[0]:42s} {r[1]:7.1f} {r[2]:7.1f} {r[3]:7.1f} {r[4]:8.1f}")
print("doc prose: 11 GB / 273 GB/s =", round(11e9 / BW * 1e3, 1), "ms/token; backbone-only FP8 8.2 GB ->", round(8.2e9 / BW * 1e3, 1), "ms/token")

print("\n== CoC token budget inside 100 ms, 2B NVFP4 student (T=2560) ==")
pre = 2 * 2e9 * 2560 / (eta * F) * 1e3
tok = 2e9 * 0.5625 / BW * 1e3
exp4 = 4 * 0.5e9 / BW * 1e3
cases = {
    "bound (doc assumptions)": (pre, tok, exp4),
    "realistic eff (decode 78%, expert 20% of BW bound; FlashDrive)": (pre, tok / 0.78, exp4 / 0.20),
    "bound + streaming KV reuse (4 of 16 images new per tick)": (pre / 4, tok, exp4),
    "realistic + streaming KV reuse": (pre / 4, tok / 0.78, exp4 / 0.20),
}
for k, (p, t, e) in cases.items():
    print(f"{k:66s} prefill {p:5.1f}  expert {e:5.1f}  ms/token {t:4.2f}  -> max CoC tokens in 100 ms: {max(0,(100-p-e)/t):5.1f}")

print("\n== Evaluation budget: plan v3 sweep = 441 scenes x 3 seeds x 200 steps = 34-85 GPU-h ==")
sweeps = {"P0 TransFuser + A1.5": 2, "P1 base + 3 rewards x 3 train seeds": 10, "P2 teacher + 2 arms x 4 stages": 9, "P3 format check (BF16/FP8/NVFP4)": 3}
tot = sum(sweeps.values())
for k, v in sweeps.items(): print(f"{k:40s} {v:3d} sweeps -> {34*v:5d}-{85*v:5d} GPU-h")
print(f"{'total':40s} {tot:3d} sweeps -> {34*tot}-{85*tot} GPU-h -> ${34*tot*1.0:,.0f}-${85*tot*1.6:,.0f} at $1.00-1.60/h (RTX PRO 6000, vast.ai)")
print("doc P1 RL: 225-450 H100-h at $1.93-2.40 ->", f"${225*1.93:,.0f}-${450*2.40:,.0f}")

print("\n== P2 gate on the continuous challenge scene_score: paired MDE = (z_a + z_b) * sd_d / sqrt(n) ==")
k1 = (Z(0.95) + Z(0.8)) / sqrt(441)
k2 = (Z(0.975) + Z(0.8)) / sqrt(441)
for sd in (0.2, 0.3, 0.4, 0.5, 0.6):
    print(f"sd_d={sd}: MDE one-sided {100*k1*sd:.2f} pp, two-sided {100*k2*sd:.2f} pp (n=441, before cluster adjustment)")
for s in (0.5, 0.7, 0.9):
    print(f"teacher mean {s}: margin {100*0.1*s:.1f} pp is decidable (one-sided, 80%) if sd_d <= {0.1*s/k1:.2f}")
