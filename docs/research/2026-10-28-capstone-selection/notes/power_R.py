#!/usr/bin/env python3
"""power_R.py — red-team arithmetic for F2 (derived quantities only; no predictions).
Run: python3 -I power_R.py > power_R.out.txt
Anchors: 03 §4 S1–S3 (MDE = (z_a+z_b)·σ_d/√n, one-sided α=0.05, power 0.8; replicate design effect 1+(m−1)ρ;
m=3, ρ=0.6 ⇒ 1.3636× scenes at equal precision). Every σ_d, ρ and ψ below is an ASSUMED input, labelled."""
import math
z = 1.6449 + 0.8416   # one-sided α=0.05, power 0.80 (03 §4)
def mde(n, sd, m=1, rho=0.6):
    neff = n*m/(1+(m-1)*rho)
    return z*sd/math.sqrt(neff)*100, neff
def n_needed(delta_pp, sd, m=3, rho=0.6):
    n1 = (z*sd/(delta_pp/100))**2            # single replicate scenes
    return n1, n1*(1+(m-1)*rho)/m             # scenes with m replicates
print("="*78); print("R1. Paired MDE (pp of scene_score) by stratum size; σ_d assumed; m=3 replicates, ρ=0.6 (03 §4 anchors)"); print("="*78)
print(f"  check anchors: n=441 σ_d=0.3 m=1 -> {mde(441,0.3)[0]:.2f} pp (03 §4: 3.55); σ_d=0.5 -> {mde(441,0.5)[0]:.2f} (5.92); m=3 ρ=0.6 n_eff/n = {3/2.2:.4f} (1/1.3636 = {1/1.3636:.4f})")
print(f"  {'scenes':>7} | " + " | ".join(f"σ_d={sd:.1f}: m=1 / m=3" for sd in (0.2,0.3,0.5)))
for n in (9, 15, 30, 60, 100, 150, 441):
    cells = []
    for sd in (0.2,0.3,0.5):
        a,_ = mde(n,sd,1); b,_ = mde(n,sd,3)
        cells.append(f"{a:5.1f} / {b:5.1f} pp")
    print(f"  {n:>7} | " + " | ".join(cells))
print("  -> a stratum of 9–30 scenes has an MDE of 8–36 pp (m=3) / 9–41 pp (m=1): any sub-10 pp per-tick loss is INSUFFICIENT by construction.")
print("\n"+"="*78); print("R2. Scenes needed per stratum to detect Δ (one-sided 80 %), σ_d assumed; m=3 ρ=0.6"); print("="*78)
print(f"  {'Δ (pp)':>7} | " + " | ".join(f"σ_d={sd:.1f}: n(m=1) / n(m=3)" for sd in (0.2,0.3,0.5)))
for d in (3,5,10,20):
    cells=[]
    for sd in (0.2,0.3,0.5):
        a,b = n_needed(d,sd); cells.append(f"{a:6.0f} / {b:6.0f}")
    print(f"  {d:>7} | " + " | ".join(cells))
print("  -> the 441 split has an unknown dynamic-actor count; the curated pool is 2,202 scenes (local_evaluation README): the only way to power a dynamic stratum is targeted sampling from the pool, not 441×3.")
print("\n"+"="*78); print("R3. If the per-scene outcome is near-binary (collision/off-road → 0), McNemar-type MDE ≈ z·√(ψ/n) with ψ = discordance (assumed)"); print("="*78)
for n in (15, 30, 60, 100, 441):
    print(f"  n={n:>4}: " + "  ".join(f"ψ={p:.2f}: {z*math.sqrt(p/n)*100:5.1f} pp" for p in (0.04,0.10,0.20)))
print(f"  (approximation; the repo's Connor anchor at ψ=4 pct, n=441 is 2.72 pp, this normal form gives {z*math.sqrt(0.04/441)*100:.2f} pp; ψ = share of scenes whose failure flips between arms)")
print("\n"+"="*78); print("R4. The E2 pilot (100 scenes × 3 reps × {1,2} ticks) seen per stratum"); print("="*78)
for frac,name in ((0.15,"dynamic-actor share 15 % (15 scenes)"),(0.30,"30 % (30 scenes)"),(1.0,"whole pilot (100)")):
    n=int(round(100*frac))
    print(f"  {name:<40} MDE σ_d=0.3: {mde(n,0.3,3)[0]:5.1f} pp; σ_d=0.5: {mde(n,0.5,3)[0]:5.1f} pp")
print("  -> F1's rule 'go if the paired loss at d=2 exceeds the replicate floor on any pre-registered stratum' is a point-estimate screen with ~15–30 pp per-stratum MDE: it cannot distinguish 5 pp from 0.")
print("\n"+"="*78); print("R5. Hours: capacity vs [C] + capstone (02 §3 rows; F1 §6/F9)"); print("="*78)
cap=(12*25,12*30); c=(306,340); add=(42,62)
print(f"  capacity {cap[0]}–{cap[1]} h; S2–S7 [C] {c[0]}–{c[1]} h; capstone +{add[0]}–{add[1]} h")
print(f"  net slack = capacity − ([C]+capstone): worst {cap[0]-(c[1]+add[1]):+d} h, best {cap[1]-(c[0]+add[0]):+d} h")
print(f"  F1/F9 claims the ~40 h [O] pool 'absorbs most of it'. [O] is NOT inside the 306–340 [C] total (02 §0.1: '[O] adds up to about +40 h'), so cutting [O] frees 0 h of [C] capacity.")
print(f"  at 25 h/wk the plan is {c[1]+add[1]-cap[0]} h over in the worst case and {c[0]+add[0]-cap[0]} h over in the best; at 30 h/wk: {c[1]+add[1]-cap[1]:+d} / {c[0]+add[0]-cap[1]:+d} h")
print(f"  02 §3 itself says 370–405 h is '12–16 weeks at 25–30 h/week' -> the 12-week frame already assumes the fast end.")
print("\n"+"="*78); print("R6. Domino and the 'CoC token decides the tick' claim (PR #45: 16 drafted / 17 verified rows, 8.30 accepted per round, greedy)"); print("="*78)
tok_ms = 1000/136.0
for acc in (8.30,):
    rounds = math.ceil(16/acc)
    print(f"  serial decode (F3 proxy): 16 tokens × {tok_ms:.2f} ms = {16*tok_ms:.0f} ms; with Domino at {acc} accepted/round: {rounds} rounds")
    for per_round in (1.5, 2.5):
        print(f"    if one verify round ≈ {per_round}× a serial token's weight read (17 rows, memory-bound) -> ≈ {rounds*per_round*tok_ms:.0f} ms for 16 tokens, ≈ {rounds*per_round*tok_ms/16:.1f} ms/token effective (assumption, labelled)")
print("  -> under NVIDIA's deployed decode, the marginal CoC cost is ~1–2.5 ms/token, the same order as the expert step (≤1.7 ms): 'n decides the band, K does not' holds for serial decode only.")
print("\n"+"="*78); print("R7. Thor-in-loop device time for E9 without a DRIVE number: bracket by what is measured on Jetson"); print("="*78)
for L,lab in ((100,"1 tick budget"),(811,"FlashRT fast 10B (Jetson, MAXN locked)"),(3322,"official BF16 10B (Jetson)")):
    print(f"  100 scenes × 3 × 200 calls at {L} ms: {100*3*200*L/1e3/3600:5.1f} h  ({lab})")
print("  -> the Jetson 2B value is unknown until T3; quote the bracket, never 171 ms (DRIVE).")
print("\nDone.")
