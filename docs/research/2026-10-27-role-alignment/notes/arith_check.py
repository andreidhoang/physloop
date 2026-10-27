# Known answers behind plan v4.1 (2026-10-27). Python stdlib only: run with `python3 -I arith_check.py`.
# Each block names the 02/03 item it serves. Inputs marked "illustrative" are not predictions,
# tolerances or margins; those stay Huy's (`____`).
from math import sqrt, exp, log, ceil, erf, pi
from statistics import NormalDist

N01 = NormalDist()
Z = N01.inv_cdf
Phi = N01.cdf


def phi(x):
    return exp(-0.5 * x * x) / sqrt(2 * pi)


print("== S5(b) flow-step ladder: 1-D Gaussian flow, X0~N(0,1) -> X1~N(2, 0.5^2), uniform Euler (03 S5 exercise) ==")
mu, s = 2.0, 0.5
beta = lambda t: (t * s * s - (1 - t)) / ((1 - t) ** 2 + (t * s) ** 2)
for K in (10, 5, 4, 2):
    g = 1.0
    for k in range(K):
        g *= 1 + beta(k / K) / K
    print(f"K={K:2d}: output std {g:.5f}, relative MSE vs exact map {100 * (g - s) ** 2 / s ** 2:.2f}%")

print("\n== S3 allocation: one replicate on more scenes vs m replicates on fewer (equal precision) ==")
print("Var(mean) = s^2/(n*m) * [1 + (m-1)*rho]; equal precision => n1 = n*m / (1 + (m-1)*rho) single-replicate scenes")
for rho in (0.3, 0.6, 0.9):
    m = 3
    n1_per_n = m / (1 + (m - 1) * rho)
    print(f"m=3, rho={rho} (illustrative): scenes x{n1_per_n:.4f}, rollouts {n1_per_n / m:.4f} of 3n -> saving {100 * (1 - n1_per_n / m):.1f}%")

print("\n== S1 intersection-union power: overall GO power = product of per-stratum powers (road-disjoint strata) ==")
print(f"4 strata at 0.80 each -> overall {0.8 ** 4:.4f}")
for k in (2, 3, 4, 6):
    print(f"{k} strata: per-stratum power needed for 0.80 overall = {0.8 ** (1 / k):.4f}")

print("\n== S2 common random numbers: sd_d^2 = sA^2 + sB^2 - 2*rho*sA*sB; continuous MDE = (z_a+z_b)*sd_d/sqrt(n) ==")
kmde = (Z(0.95) + Z(0.8)) / sqrt(441)
for rho in (0.0, 0.5, 0.8):
    sa = sb = 0.4  # illustrative
    sd = sqrt(sa * sa + sb * sb - 2 * rho * sa * sb)
    print(f"sA=sB=0.4, rho={rho}: sd_d={sd:.4f} -> one-sided 80% MDE at n=441: {100 * kmde * sd:.2f} pp")
print(f"check vs 03 S3 row: sd_d=0.3 -> {100 * kmde * 0.3:.2f} pp; sd_d=0.5 -> {100 * kmde * 0.5:.2f} pp")


def bvn_lower(a, b, r, n=4000, lo=-9.0):
    """P(Z1 < a, Z2 < b) for a standard bivariate normal with correlation r (Simpson on z1)."""
    h = (a - lo) / n
    tot = 0.0
    for i in range(n + 1):
        z = lo + i * h
        w = 1 if i in (0, n) else (4 if i % 2 else 2)
        tot += w * phi(z) * Phi((b - r * z) / sqrt(1 - r * r))
    return tot * h / 3


def solve(f, lo, hi, it=80):
    for _ in range(it):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


print("\n== S1 group-sequential tables, K=2 equally spaced looks (corr sqrt(1/2)), one-sided ==")
r = sqrt(0.5)
for alpha in (0.05, 0.025):
    # O'Brien-Fleming: c1 = c*sqrt(2), c2 = c
    c = solve(lambda c: (1 - bvn_lower(c * sqrt(2), c, r)) - alpha, 0.5, 5.0)
    # Pocock: c1 = c2 = c
    p = solve(lambda c: (1 - bvn_lower(c, c, r)) - alpha, 0.5, 5.0)
    print(f"alpha={alpha}: OBF z-boundaries {c * sqrt(2):.4f}, {c:.4f} (fixed-n {Z(1 - alpha):.4f}); Pocock {p:.4f}, {p:.4f}")

print("\n== S4 GRPO: group-standardized advantages are invariant to a positive affine reward change ==")
def adv(rw):
    m = sum(rw) / len(rw)
    sd = sqrt(sum((x - m) ** 2 for x in rw) / len(rw))
    return [(x - m) / sd for x in rw]
base = [1, 0, 0, 1, 1, 0, 0, 0]
for name, rw in (("[1,0,0,1,1,0,0,0] (03 S4)", base),
                 ("collision-only group: -10 x3, 0 x5", [-10, -10, -10, 0, 0, 0, 0, 0]),
                 ("same events at -1", [-1, -1, -1, 0, 0, 0, 0, 0]),
                 ("affine 7*r - 3 of the 03 group", [7 * x - 3 for x in base])):
    a = adv(rw)
    print(f"{name:40s} -> advantages {sorted(set(round(x, 5) for x in a))}")
print("all-equal group (e.g. n_generation=1, or every rollout scores the same): advantage = 0/0 -> 0 by convention; no gradient")

print("\n== S4 AlpaGym ratio: clip on the geometric-mean ratio vs the joint ratio (5 SDE steps x 64 waypoints x 2 controls) ==")
n_el = 5 * 64 * 2
for lo_hi in (0.8, 1.2):
    print(f"mean-ratio bound {lo_hi} -> joint log-ratio {n_el * log(lo_hi):+.1f} (joint ratio e^{n_el * log(lo_hi):+.1f})")
print("gradient of the mean-ratio objective w.r.t. one element's log-prob is scaled by 1/n_el =", f"1/{n_el} = {1 / n_el:.5f}")

print("\n== S4 KL: closed-form Gaussian KL vs the k3 estimator ==")
d_mu, sig = 0.1, 0.5  # illustrative per-element mean shift and shared SDE std
print(f"KL(N(mu1,s^2)||N(mu2,s^2)) = (mu1-mu2)^2/(2 s^2) = {d_mu ** 2 / (2 * sig ** 2):.4f} for dmu={d_mu}, s={sig}")
for rr in (0.8, 1.25):
    print(f"k3 at r={rr}: r-1-ln r = {rr - 1 - log(rr):.5f}")

print("\n== S2/S4 performance: Amdahl bound for one lever, S = 1/((1-f) + f/k) ==")
for f in (0.5, 0.75, 0.875):  # illustrative shares of step time
    print(f"f={f}: k=2 -> {1 / ((1 - f) + f / 2):.3f}x; k=4 -> {1 / ((1 - f) + f / 4):.3f}x; k->inf -> {1 / (1 - f):.3f}x")

print("\n== S5/P-012 scaling: all-reduce bus bandwidth factor busbw = algbw * 2(n-1)/n; strong-scaling efficiency ==")
for n in (2, 4, 8, 16):
    print(f"n={n:2d}: busbw/algbw = {2 * (n - 1) / n:.4f}")
T1, T8 = 100.0, 16.0  # illustrative step times
print(f"E(8) = T1/(8*T8) = {T1 / (8 * T8):.4f} for T1={T1}s, T8={T8}s")

print("\n== S7 delay injection is quantized to the control step (assumes the buffer is read at control ticks) ==")
dt = 100
for d in (139, 171, 200):
    print(f"delay {d} ms at dt=100 ms -> applied {ceil(d / dt)} ticks late; at 15 m/s the car moves {15 * d / 1000:.3f} m meanwhile")

print("\n== Costs at the 2026-10-27 price basis (vast.ai on-demand; Lambda/RunPod/Nebius list) ==")
sweep_lo, sweep_hi = 441 * 3 * 200 * 0.463 / 3600, 441 * 3 * 200 * 1.150 / 3600
print(f"one sweep (single-stream bound): {sweep_lo:.1f}-{sweep_hi:.1f} GPU-h -> ${sweep_lo * 1.00:.0f}-${sweep_hi * 1.56:.0f} at RTX PRO 6000 WS $1.00-1.56")
print(f"S3 teacher reuse at an illustrative 20% mini-control: saves {0.8 * sweep_lo:.1f}-{0.8 * sweep_hi:.1f} GPU-h per reused teacher sweep; both formats {1.6 * sweep_lo:.1f}-{1.6 * sweep_hi:.1f}")
print(f"S5 one 8xH100 node at $31.92-36.00/h x 24-48 h: ${31.92 * 24:,.0f}-${36.00 * 48:,.0f}")
pr42 = 1.77 * 100_000 * 128 / 3600
print(f"PR #42 stage 1 at 1.77 s/step x 100k steps x 128 GPUs = {pr42:,.0f} H100-h -> ${pr42 * 1.93:,.0f}-${pr42 * 2.40:,.0f} at $1.93-2.40; 10% pilot {pr42 / 10:,.0f} H100-h")
print(f"S4 training 225-450 H100-h at $1.79-2.40 (10-27): ${225 * 1.79:,.0f}-${450 * 2.40:,.0f}")
