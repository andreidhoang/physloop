#!/usr/bin/env python3
"""arith_F.py — every number in the PhysLoop capstone selection (F1 analysis, F2 red-team rulings, final report).

Run:  python3 -I arith_F.py  > arith_F.out.txt

Rules (brief §Method 6):
- Device constants come only from Track D (IDs in brackets).
- Model shapes come only from the repo or the research notes (IDs in brackets).
- Every roofline number is a BOUND ("inferred, at 100% of peak"), never a prediction.
- Nothing here is measured by PhysLoop. Measured numbers quoted from sources are labelled "reported".
Revision 2 (2026-10-29): applies the red team's accepted fixes — F5 Jetson bracket (no DRIVE stand-in), F9 corrected slack,
F10 re-score (final columns beside F1's), F12 redesigned pilot costs, F13 per-stratum power (ported from R1/R2/R4), F14 Domino regime.
"""
import math

# ----------------------------------------------------------------------------
# 0. Device constants (Track D only)
# ----------------------------------------------------------------------------
THOR_BW = 273e9            # B/s, LPDDR5X, shared with CPU/ISP [D-DT1, D-DT7]
THOR_FP8_MAXN = 517e12     # dense FP8 FLOP/s, MAXN [D-DT2]
THOR_BF16_MAXN = 258.5e12  # derived = FP8/2 [D-DT2, d_arith]
THOR_FP4_MAXN = 1035e12    # dense FP4 [D-DT2]
THOR_120W_SCALE = 0.880    # GPU 1,386 vs 1,575 MHz [D-DT3, d_arith]
THOR_SMS = 20              # user runtime query [D-DT9]
THOR_L2 = 32 * 2**20       # 32 MiB, user runtime query [D-DT9]
THOR_TDP_W = 130.0         # dev-kit TDP [D-DT3]; default mode 120 W [D-DT3]
RTX6000_BW = 1.792e12      # RTX PRO 6000 WS [D-DP1]
RTX6000_FP8 = 1000e12      # derived 4:2:1 from 4,000 sparse FP4 [D-DP1, F3.2]
RTX6000_BF16 = 500e12
RTX6000_SMS = 188          # 24,064/128 [D-F3.2]
# Prices, vast.ai on-demand 2026-10-28 16:15:50 UTC, cheapest / per-offer median $/GPU-h [D-DR1]
P_WS = (1.00, 1.47)
P_SERVER = (1.24, 1.38)
P_H100 = (1.93, 3.14)
P_B200 = (7.50, 8.75)
P_MIG24 = (0.50, 0.69)     # RunPod RTX PRO 6000 MIG 24 GB, page updated 2026-10-08 [D-DR3]
P_8XH100_VAST = 29.33      # vast.ai's one 8 x H100 SXM offer, $/h for 8 GPUs [D-F6.3]
P_8XH100_LIST = (31.92, 36.00)  # list, Lambda/RunPod/Nebius x 8 [02 §5]
JETSON_MSRP = 5499.0       # [D-DT5]; SabrePC $6,268.86 out of stock [D-DR6]

# ----------------------------------------------------------------------------
# 1. Model shapes (repo + notes only)
# ----------------------------------------------------------------------------
# Student VLM backbone proxy: Qwen3-VL-2B text config [A-X.2]
S_NONEMB = 1.409e9; S_EMB = 0.311e9; S_LAYERS = 28; S_HID = 2048; S_KVH = 8; S_HD = 128; S_INTER = 6144; S_VOCAB = 151936
# Compact expert, PR #42 [L5 1.3 / 1.3i]: 28 layers, hidden 384, 16 heads x 128, 8 KV heads, inter 1,536.
# Two derived parameter counts exist: 0.116e9 (L5 1.3i) and ≈ 0.050e9 (E §7.7, 12·d²·L); neither checked against PR #42's code (unverified).
E_PARAMS = 0.116e9; E_PARAMS_LOW = 0.050e9
KV_BYTES_PER_TOKEN_STUDENT = S_LAYERS * S_KVH * S_HD * 2 * 2   # BF16 K and V [L5 1.3i]
# Teacher Alpamayo 1.5 [04 §2, §4]: VLM KV 147,456 B/token; expert 2.279e9 params; 5.0 GB/step BF16 (repo figure)
KV_BYTES_PER_TOKEN_TEACHER = 147456
T_EXPERT_PARAMS = 2.279e9
T_EXPERT_STEP_BYTES_BF16 = 5.0e9
# Request shapes, PR #45 [A-Q2.2]: 4 cams x 4 frames 576x320; newest frame only encoded; 3,006 prompt rows;
# 6 candidates x 64 steps; K = 5; fixed noise; ~10 ms host. Batch size is NOT stated in the README (F2 row 13); B = 1 is Track A's reading.
T_PROMPT = 3006; VIS_TOK_PER_IMG = 180; N_IMG_NEWEST = 4; N_IMG_ALL = 16; S_CAND = 6; A_STEPS = 64; K_DEPLOYED = 5
DT_MS = 100.0  # AlpaSim control step [E §3.3]

# Measured proxies (reported, not PhysLoop's): Edge-LLM v0.10.0, Qwen3-VL-2B-Instruct, Jetson AGX Thor dev kit,
# JetPack 7.2 / TRT 10.16, NVFP4 LLM / FP16 ViT, B = 1, power mode NOT stated [A-Q1.10, A-J6, D-F2.4; F2 row 10 re-opened]
M_PREFILL_2048_MS = 28.0; M_DECODE_MS_PER_TOK = 1000.0 / 136.0; M_VIT_265_MS = 11.4
# FlashRT on Jetson AGX Thor, Alpamayo 1.5-10B, MAXN locked clocks, `balanced` stage split (one clip, seed 0) [A-Q4.3; F2 row 11]
FR_VISION = 121.0; FR_PREFILL = 246.0; FR_DECODE_PER_TOK = 22.1; FR_EXPERT_K10 = 126.0; FR_FAST_TOTAL = 811.0; FR_OFFICIAL = 3322.0; FR_STREAM = 345.0
# PR #45 on DRIVE AGX Thor (never a Jetson number) [A-Q2.2]
PR45_FP8 = 171.2; PR45_FP16 = 220.6; PR45_HOST_MS = 10.0; PR45_ACCEPTED_PER_ROUND = 8.30; PR45_DRAFT = 16
# FlashDrive on RTX PRO 6000 (edition unstated) [A-Q5.3]
FD_BASE = 716.9; FD_SYS = 513.3; FD_ALL = 151.4; FD_ACTION_SYS = 113.9; FD_ACTION_BASE = 192.9

def ms(bytes_, bw): return bytes_ / bw * 1e3
def ticks(L_ms, dt=DT_MS): return math.ceil(L_ms / dt - 1e-12) if L_ms > 0 else 0

print("=" * 78)
print("F0. Tension 1 — expert efficiency: sm_120 (FlashDrive-derived) vs Jetson (FlashRT stage times)")
print("=" * 78)
b_ws = ms(T_EXPERT_STEP_BYTES_BF16, RTX6000_BW)
print(f"  RTX PRO 6000 WS, BF16 teacher expert: bound {b_ws:.2f} ms/step (5.0 GB/1.792 TB/s) -> 8 steps {8*b_ws:.1f} ms;"
      f" FlashDrive sys-opt action 113.9 ms -> {8*b_ws/FD_ACTION_SYS*100:.1f}% of bound; baseline row 192.9 -> {8*b_ws/FD_ACTION_BASE*100:.1f}% [04 §4, P-007]")
fr_step = FR_EXPERT_K10 / 10
b_thor_w = ms(T_EXPERT_PARAMS * 1, THOR_BW)
b_thor_wkv = ms(T_EXPERT_PARAMS * 1 + 3000 * KV_BYTES_PER_TOKEN_TEACHER, THOR_BW)
print(f"  Jetson AGX Thor, FlashRT balanced FP8 expert: {fr_step:.2f} ms/step reported; bound weights-only {b_thor_w:.2f} ms,"
      f" +BF16 KV(3k) {b_thor_wkv:.2f} ms -> {b_thor_w/fr_step*100:.0f}%–{b_thor_wkv/fr_step*100:.0f}% of bound [A-Q4.6]")
print("  -> Different device, precision and kernels; both inferred. For the Thor question Track A wins (target device).")
print("     The '20%' premise of P-007 is an sm_120/FlashDrive derivation; it is a sweep-cost question, not a Thor lever.")

print("\n" + "=" * 78)
print("F1. Tension 2 — tick quantization ⌈L/Δt⌉ (AlpaSim DelayBuffer read once per control tick) [E §3.1–3.2; F2 rows 1–2 re-opened]")
print("=" * 78)
v = 15.0
print(f"  {'L (ms)':>8} {'ticks':>6} {'applied ms':>11} {'v*L (m)':>9} {'applied (m)':>12}")
for L in [0, 1, 50, 99, 100, 101, 139, 171, 200, 201, 300, 345, 500, 811, 3322]:
    t = ticks(L); print(f"  {L:>8} {t:>6} {t*DT_MS:>11.0f} {v*L/1e3:>9.3f} {v*t*DT_MS/1e3:>12.3f}")
print("  -> 139 ms is SIMULATED as 200 ms of plan staleness (3.0 m at 15 m/s); 2.085 m is the physical distance during 139 ms.")
print("     Track E (code at named commits, identical on main and e2e_challenge) wins over the S7 card / 03 §4 text.")
print("  Bench2Drive-Robust uses floor (SIM_RATE inferred 20 Hz, F2 row 14): 100 ms -> floor(100*20/1000) = 2 ticks of 50 ms = 100 ms applied;")
print(f"     AlpaSim: 100 ms -> {ticks(100)} tick of 100 ms = 100 ms applied; 139 ms -> {ticks(139)} ticks = 200 ms applied. Compare in applied-staleness seconds, never nominal ms.")

print("\n" + "=" * 78)
print("F2. Tension 3 — 2B decode: bound vs Edge-LLM's measured 7.35 ms/token on Jetson AGX Thor (Qwen3-VL-2B proxy)")
print("=" * 78)
for label, bpp_body, head_bytes, kv_tok in [
    ("C11/03 §4 style: 2.0B @ 4.5 bpp, no head split, no KV", None, None, 0),
    ("NVFP4 body (4.5 bpp) + NVFP4 head + BF16 KV @2,048", 4.5, S_EMB * 4.5 / 8, 2048),
    ("NVFP4 body (4.5 bpp) + FP16 head + BF16 KV @2,048", 4.5, S_EMB * 2, 2048),
    ("NVFP4 body (4.5 bpp) + FP16 head + BF16 KV @3,006", 4.5, S_EMB * 2, 3006),
]:
    by = 2.0e9 * 4.5 / 8 if bpp_body is None else S_NONEMB * bpp_body / 8 + head_bytes + kv_tok * KV_BYTES_PER_TOKEN_STUDENT
    b = ms(by, THOR_BW)
    print(f"  {label:<58} {by/1e9:5.2f} GB -> {b:5.2f} ms/token (100% of 273 GB/s); measured 7.35 -> {b/M_DECODE_MS_PER_TOK*100:4.0f}% of bound")
print("  -> Measured (reported) is the planning figure; the bound is a floor whose efficiency reading depends on the unstated LM-head precision [A-X.2].")

print("\n" + "=" * 78)
print("F3. Student per-stage bounds and measured proxies on Jetson AGX Thor; L(n) under SERIAL decode (proposed structure; the latency model is ★)")
print("=" * 78)
fl_lin = 2 * S_NONEMB * T_PROMPT
fl_attn = 4 * S_LAYERS * T_PROMPT**2 * S_HID
for mode, scale in [("MAXN", 1.0), ("120 W", THOR_120W_SCALE)]:
    pf8 = (fl_lin + fl_attn) / (THOR_FP8_MAXN * scale) * 1e3
    pf4 = (fl_lin / (THOR_FP4_MAXN * scale) + fl_attn / (THOR_FP8_MAXN * scale)) * 1e3
    print(f"  prefill {T_PROMPT} rows, {mode}: {fl_lin/1e12:.2f} TFLOP linear + {fl_attn/1e12:.2f} TFLOP attention -> FP8 bound {pf8:.1f} ms; NVFP4-linear bound {pf4:.1f} ms")
pf_meas = M_PREFILL_2048_MS * T_PROMPT / 2048
print(f"  prefill measured proxy (Edge-LLM NVFP4, 2,048 rows 28.0 ms, linear scaling): {pf_meas:.1f} ms  [A-Q1.10; power mode unstated]")
vit_newest = M_VIT_265_MS / 265 * VIS_TOK_PER_IMG * N_IMG_NEWEST
vit_all = M_VIT_265_MS / 265 * VIS_TOK_PER_IMG * N_IMG_ALL
print(f"  ViT measured proxy (FP16 ViT, 265 tokens 11.4 ms, linear): newest 4 frames ({VIS_TOK_PER_IMG*N_IMG_NEWEST} tok) {vit_newest:.1f} ms; all 16 frames ({VIS_TOK_PER_IMG*N_IMG_ALL} tok) {vit_all:.1f} ms")
M_rows = S_CAND * A_STEPS
kv_bf16 = T_PROMPT * KV_BYTES_PER_TOKEN_STUDENT
for ep, el in [(E_PARAMS, "0.116e9 (L5 1.3i)"), (E_PARAMS_LOW, "0.050e9 (E §7.7)")]:
    st = ms(ep * 1 + kv_bf16, THOR_BW); st2 = ms(ep * 1 + kv_bf16 / 2, THOR_BW)
    print(f"  expert step bound, FP8 weights at {el}: +BF16 KV {st:.2f} ms/step (K=5 {5*st:.1f} ms); +FP8 KV {st2:.2f} ms/step (K=5 {5*st2:.1f} ms)")
fl_exp = 2 * E_PARAMS * M_rows + 2 * 2 * S_LAYERS * M_rows * T_PROMPT * S_HID
print(f"  expert step FLOPs (M={M_rows}): {fl_exp/1e9:.0f} GFLOP -> {fl_exp/THOR_FP8_MAXN*1e3:.2f} ms at 517 TFLOPS: memory-bound (bytes dominate)")
print(f"  weights {E_PARAMS/1e6:.0f} M FP8 = {E_PARAMS/2**20:.0f} MiB > 32 MiB L2 -> re-read from DRAM every step on Thor [D-DT9, D-F4.3]")
host = PR45_HOST_MS
exp5_b = 5 * ms(E_PARAMS * 1 + kv_bf16, THOR_BW)
print("\n  L(n) = ViT + prefill + n·decode + K·expert + host (ms); 'bound' at 100% of peak, 'proxy' from Edge-LLM's measured Qwen3-VL-2B on Jetson:")
rows = {
    "bound: ViT excluded (cached) + prefill FP8 bound + expert K=5 bound + host 10": (0 + (fl_lin + fl_attn) / THOR_FP8_MAXN * 1e3 + exp5_b + host, ms(S_NONEMB*4.5/8 + S_EMB*4.5/8 + 3006*KV_BYTES_PER_TOKEN_STUDENT, THOR_BW)),
    "proxy: ViT newest-4 + prefill proxy + expert K=5 bound + host 10": (vit_newest + pf_meas + exp5_b + host, M_DECODE_MS_PER_TOK),
    "proxy: ViT newest-4 + prefill proxy + expert K=5 at 2x bound + host 10": (vit_newest + pf_meas + 2*exp5_b + host, M_DECODE_MS_PER_TOK),
    "proxy: ViT all-16 + prefill proxy + expert K=5 bound + host 10": (vit_all + pf_meas + exp5_b + host, M_DECODE_MS_PER_TOK),
}
for name, (L0, per_tok) in rows.items():
    n1 = max(0.0, (100 - L0) / per_tok); n2 = max(0.0, (200 - L0) / per_tok)
    print(f"  {name}")
    print(f"      L0 = {L0:6.1f} ms at n = 0 ({ticks(L0)} tick); per token {per_tok:.2f} ms; L crosses 100 ms at n = {n1:4.1f}, 200 ms at n = {n2:4.1f};"
          f" n=16: {L0+16*per_tok:.0f} ms ({ticks(L0+16*per_tok)} ticks)")
print("  -> Under SERIAL decode the marginal term is the CoC token (7.35 ms measured, 3.5–6.4 ms bound), not the expert step (≤ 1.7 ms bound). See F14 for Domino.")
print(f"  PR #45 on DRIVE AGX Thor (reported, never Jetson): FP8 {PR45_FP8} ms = {ticks(PR45_FP8)} ticks; FP16 {PR45_FP16} ms = {ticks(PR45_FP16)} ticks; cut needed to reach 100 ms: {PR45_FP8-100:.1f} ms = {(PR45_FP8-100)/PR45_FP8*100:.1f}%")

print("\n" + "=" * 78)
print("F4. Teacher (Alpamayo 1.5-10B) tick positions on each device (reported latencies -> ticks)")
print("=" * 78)
for name, L in [("Jetson official BF16 (FlashRT baseline arm)", FR_OFFICIAL), ("Jetson FlashRT fast", FR_FAST_TOTAL),
                ("Jetson FlashRT streaming (fine-tuned weights, 8 steps)", FR_STREAM), ("RTX PRO 6000 FlashDrive baseline", FD_BASE),
                ("RTX PRO 6000 FlashDrive sys-opt", FD_SYS), ("RTX PRO 6000 FlashDrive all+W4A8", FD_ALL)]:
    print(f"  {name:<58} {L:7.1f} ms -> {ticks(L):2d} ticks late at Δt = 100 ms")
dec16 = 16 * FR_DECODE_PER_TOK; k_half = 5 * (FR_EXPERT_K10 / 10)
print(f"  Jetson balanced: 16 CoC tokens = {dec16:.0f} ms; K 10->5 saves {k_half:.0f} ms; fast 811 -> no-CoC ≈ {FR_FAST_TOTAL-16*20.8:.0f} ms ({ticks(FR_FAST_TOTAL-16*20.8)} ticks); K=5 ≈ {FR_FAST_TOTAL-k_half:.0f} ms ({ticks(FR_FAST_TOTAL-k_half)} ticks)")
print("  -> On Jetson the 10B sits at 8–9 ticks; its levers move whole ticks only far from the 1–2-tick regime that the 2B question lives in.")

print("\n" + "=" * 78)
print("F5. Sweep costs and Thor-in-loop device time (steps x s/step; prices DR1 2026-10-28; s/step is FlashDrive's 10B figure incl. rendering, ±2x until S2's profile)")
print("=" * 78)
STEPS_SWEEP = 441 * 3 * 200
SPS = (0.463, 1.150)   # FlashDrive per-step rollout time, optimized / baseline [A-Q5.6, 04 §4]
def gpuh(steps): return (steps * SPS[0] / 3600, steps * SPS[1] / 3600)
sw = gpuh(STEPS_SWEEP)
print(f"  one sweep: {STEPS_SWEEP:,} steps -> {sw[0]:.1f}–{sw[1]:.1f} GPU-h -> ${sw[0]*P_WS[0]:.0f}–${sw[1]*P_WS[1]:.0f} on RTX PRO 6000 WS")
curve3 = gpuh(3 * STEPS_SWEEP)
print(f"  whole-set curve, 3 new arms {{1,2,3 ticks}} + S2's 0 arm: {curve3[0]:.0f}–{curve3[1]:.0f} GPU-h -> ${curve3[0]*P_WS[0]:.0f}–${curve3[1]*P_WS[1]:.0f}; with the expert graph (-11.3..-13.6% [L7-2]): {curve3[0]*0.887:.0f}–{curve3[1]*0.864:.0f} GPU-h")
print("  Thor-in-loop serial device time (Jetson bracket; no DRIVE number is used; the 2B Jetson value is unknown until T3):")
for L, lab in [(100, "1-tick budget"), (811, "FlashRT fast, 10B, Jetson MAXN locked [A-Q4.3]"), (3322, "official BF16, 10B, Jetson [A-Q4.3]")]:
    print(f"    per full sweep at {L} ms/call: {STEPS_SWEEP*L/1e3/3600:5.1f} h; per 100-scene x 3 subset: {100*3*200*L/1e3/3600:5.1f} h  ({lab})")

print("\n" + "=" * 78)
print("F6. Paired MDE, whole set (continuous scene_score; one-sided α = 0.05, power 0.8) [03 §4 S3]")
print("=" * 78)
z = 1.6449 + 0.8416
def mde(n, sd, m=1, rho=0.6):
    neff = n * m / (1 + (m - 1) * rho); return z * sd / math.sqrt(neff) * 100, neff
for n in [100, 441]:
    for sd in [0.3, 0.5]:
        a, _ = mde(n, sd, 1); b, neff = mde(n, sd, 3)
        print(f"  n = {n:3d} scenes, σ_d = {sd}: MDE {a:5.2f} pp (1 replicate); with 3 replicates at ρ = 0.6 (n_eff = {neff:.0f}) {b:5.2f} pp")

print("\n" + "=" * 78)
print("F7. Batched bitwise driver (P-015 re-cut): Amdahl with invariance cost")
print("=" * 78)
def amdahl(f, s): return 1 - ((1 - f) + f / s)
for f in [0.73, 0.875]:
    for s in [2.0, 2.6]:
        base = amdahl(f, s)
        for inv in [1.34, 1.62]:   # batch-invariant kernel slowdown, LLM decode workloads [C-37, C-36i]
            s2 = s / inv
            print(f"  f = {f:.3f}, batch s = {s:.1f}: saving {base*100:4.1f}%/sweep; with invariant kernels at {inv}x: s' = {s2:.2f} -> {amdahl(f, s2)*100:4.1f}%/sweep")
eq = gpuh(STEPS_SWEEP)
print(f"  equivalence sweep (non-invariant arm, one known-null pair): {eq[0]:.0f}–{eq[1]:.0f} GPU-h -> ${eq[0]*P_WS[0]:.0f}–${eq[1]*P_WS[1]:.0f}")
sweeps_s3s5 = (21, 22)   # plan: S3 3–4 + S4 ≤ 10 + S5 ~8 [L7 §1; F2 §3]
for sv, lab in [(amdahl(0.73, 2.0/1.62), "invariant, worst"), (amdahl(0.875, 2.6), "non-invariant, best")]:
    lo = sweeps_s3s5[0] * sw[0] * sv; hi = sweeps_s3s5[1] * sw[1] * sv
    print(f"  saving on S3–S5 ({sweeps_s3s5[0]}–{sweeps_s3s5[1]} sweeps, only if all run), {lab}: {lo:.0f}–{hi:.0f} GPU-h -> ${lo*P_WS[0]:.0f}–${hi*P_WS[1]:.0f}")
print("  Batching bound (04 §4, L7-4): per-sample ≈ 253–276 ms vs 513–717 ms at B = 1 -> s ≈ 2.0–2.6 (inference, teacher on RTX PRO 6000; unmeasured).")

print("\n" + "=" * 78)
print("F8. Jetson artifact: J/tick bound, proxy choices, purchase economics")
print("=" * 78)
print(f"  J/tick upper bound at a 100 ms tick: 120 W mode -> {120*0.1:.0f} J; 130 W TDP -> {THOR_TDP_W*0.1:.0f} J; 70 W mode -> {70*0.1:.0f} J  (module power x tick; measured J/tick needs tegrastats/INA) [D-DT3]")
print(f"  Thor dev kit amortized over 2,000 h (assumption): ${JETSON_MSRP/2000:.2f}/h vs RTX PRO 6000 WS ${P_WS[0]}–${P_WS[1]}/h [D-DT5, D-DR1]")
print(f"  B200 (tcgen05 ISA proxy) ${P_B200[0]}–${P_B200[1]}/h: 10 h of ISA validation = ${10*P_B200[0]:.0f}–${10*P_B200[1]:.0f}; RTX PRO 6000 MIG 24 GB (bandwidth-side proxy) ${P_MIG24[0]}–${P_MIG24[1]}/h [D-F4.9]")
print(f"  Thor FP8 ridge {THOR_FP8_MAXN/THOR_BW:.0f} FLOP/B vs RTX PRO 6000 WS {RTX6000_FP8/RTX6000_BW:.0f}: an operator at 300–900 FLOP/B is compute-bound on the proxy, memory-bound on Thor [D-F4.2]")
print(f"  skinny GEMM M = {M_rows}, N tiles of 128: N = 2,048 -> 16 CTAs = {16/RTX6000_SMS*100:.1f}% of 188 SMs vs {16/THOR_SMS*100:.0f}% of 20 SMs [D-F4.1]")

print("\n" + "=" * 78)
print("F9. Calendar and hours (11-02 -> 01-25) [E §5, 02 §3] — corrected per F2 fix 1 ([O] is outside the [C] total)")
print("=" * 78)
cap = (12 * 25, 12 * 30); c_s2s7 = (306, 340)
adds = {"E0–E5 harness, CPU tests, positive control, targeted sampling": (8, 10),
        "P-021 sub-tick/trace-driven DelayBuffer + command-FIFO option (AlpaSim PR)": (8, 12),
        "#1 floor: eval-driver port of AlpaGym-style batching (L7-4) + invariance arm": (16, 22),
        "#3 x86 SM120 runtime (L7-6, was [G]; deferrable)": (12, 20)}
lo = sum(v[0] for v in adds.values()); hi = sum(v[1] for v in adds.values())
for k, v in adds.items(): print(f"    {k:<78} +{v[0]}–{v[1]} h")
print(f"  capacity {cap[0]}–{cap[1]} h; S2–S7 [C] {c_s2s7[0]}–{c_s2s7[1]} h; capstone +{lo}–{hi} h")
print(f"  net slack = capacity − ([C] + capstone): worst {cap[0]-(c_s2s7[1]+hi):+d} h, best {cap[1]-(c_s2s7[0]+lo):+d} h")
print(f"  at 25 h/week: {c_s2s7[0]+lo-cap[0]}–{c_s2s7[1]+hi-cap[0]} h over; at 30 h/week: {c_s2s7[0]+lo-cap[1]:+d}–{c_s2s7[1]+hi-cap[1]:+d} h over")
print("  [O] items (~40 h) sit outside the [C] total (02 §0.1), so cutting them frees 0 h of [C] capacity; S8 (40 h) is outside the window.")
disp = {"L2-1 S4 scaling model + multi-config sweep (D1)": (12, 18), "L6-6 S5 scaling and resiliency ladder (D4)": (10, 14),
        "L5-5 manifest-driven pipeline metrics (D2)": (8, 12), "L6-3 DCGM + goodput dashboard (D4)": (8, 11), "L6-2 NVIDIA-style throughput measurement (D4, D2)": (6, 8),
        "defer #3's x86 half (the capstone's own deferrable item)": (12, 20)}
dlo = sum(v[0] for v in disp.values()); dhi = sum(v[1] for v in disp.values())
for k, v in disp.items(): print(f"    displaceable [C] role item: {k:<70} {v[0]}–{v[1]} h")
print(f"  named displaceable pool: {dlo}–{dhi} h -> covers the 30 h/week overrun; at 25 h/week the window must also extend by {math.ceil((c_s2s7[0]+lo-cap[0]-dhi)/25) if c_s2s7[0]+lo-cap[0]-dhi>0 else 0}–{math.ceil((c_s2s7[1]+hi-cap[0]-dlo)/25)} weeks (02 §3 itself says 370–405 h is 12–16 weeks)")

print("\n" + "=" * 78)
print("F10. Candidate scoring (1–5 on the brief's nine criteria): F1 columns and FINAL columns after the red-team rulings; equal and weighted")
print("=" * 78)
crit = ["AV", "Gap", "Small", "Win", "Behav", "Legib", "Upstr", "Role", "Cost"]
W = {"AV": 1, "Gap": 2, "Small": 1.5, "Win": 1, "Behav": 1.5, "Legib": 1, "Upstr": 1, "Role": 1.5, "Cost": 1.5}
# (F1 scores, FINAL scores). Behaviour is scored neutral (3) where the criterion does not apply (measurement studies); 5 only for a bitwise speed-up.
cands = {
 "K22 COMPOSITE: eval-driver bitwise batching (floor) + latency instrument layer + P-021; curve conditional": ([4,3,5,4,5,4,5,5,5], [4,3,5,4,5,4,5,5,5]),
 "K2  Bitwise-invariant batched eval driver (P-015 re-cut; ports AlpaGym-style batching)":              ([4,4,5,4,5,4,5,5,4], [4,3,5,4,5,4,4,5,5]),
 "K14 AlpaSim sub-tick/trace-driven DelayBuffer + command-FIFO option (PR)":                             ([4,4,5,3,5,4,5,4,5], [4,4,5,3,3,4,5,4,5]),
 "K1  Score-per-tick latency-cost curve (S7 injection re-cut; S5(b)/(c) as lever arms)":               ([5,4,5,4,5,5,4,5,5], [5,3,3,3,3,5,4,4,4]),
 "K21 Deployment-lever verdicts at a fixed tick band (S5(b)/(c) + FlashDrive levers, merged)":         ([5,3,4,3,4,5,3,4,4], [5,3,4,3,3,5,3,4,4]),
 "K11 S7 as written (constant arms + HIL + C++ server; P-002)":                                          ([5,4,3,3,5,5,4,5,3], [5,3,3,3,3,5,4,5,3]),
 "K4  S5(c) CoC-budget curve as written":                                                                ([5,4,3,3,4,5,3,4,3], [5,4,3,3,3,5,3,4,3]),
 "K5  S5(b) flow-step ladder as written":                                                                ([4,3,5,3,5,4,3,3,5], [4,3,5,3,3,4,3,3,5]),
 "K15 Per-lever paired closed-loop re-run of FlashDrive's public bundle on 1.5":                        ([4,3,5,3,4,4,3,4,4], [4,3,5,3,3,4,3,4,4]),
 "K8  P-007 expert diagnosis E-K0 (sm_120)":                                                             ([3,2,4,3,5,3,3,3,5], [3,2,4,3,5,3,3,3,5]),
 "K7  S6 G6 per-format closed-loop verdict on real Jetson engines":                                     ([4,4,2,3,5,4,3,4,2], [4,4,2,3,4,4,3,4,2]),
 "K16 DART visual-token pruning closed-loop verdict":                                                    ([3,3,4,3,3,4,3,3,4], [3,3,4,3,3,4,3,3,4]),
 "K10 P-019 one C++ driver for two targets, as written":                                                ([4,3,3,3,3,3,3,4,3], [4,3,3,3,3,3,3,4,3]),
 "K3  Jetson AGX Thor measurement artifact + C++ driver (S6/P-019/P-008 merged)":                       ([4,4,3,3,5,4,3,4,3], [4,3,2,3,4,4,3,3,2]),
 "K18 Proxy-to-Thor transfer study":                                                                     ([3,4,2,3,5,4,3,3,2], [3,4,2,3,3,4,3,3,2]),
 "K13 sm_110 flow-expert kernel (tcgen05/TMEM)":                                                        ([4,3,2,2,4,3,3,3,2], [4,3,2,2,4,3,3,3,2]),
 "K19 Speculative-CoC paired closed-loop verdict on the student":                                        ([4,2,3,3,3,3,2,3,3], [4,2,3,3,3,3,2,3,3]),
 "K6  S6 as written: 2B at 10 Hz on Jetson with a behaviour-preserving win":                            ([5,2,2,2,3,4,2,3,2], [5,2,2,2,3,4,2,3,2]),
 "K9  P-018 few-step expert distillation":                                                               ([3,2,3,3,2,3,2,3,3], [3,2,3,3,2,3,2,3,3]),
 "K12 S8 HCMC shadow-mode OOD set":                                                                      ([3,4,1,2,5,3,2,2,1], [3,4,1,2,3,3,2,2,1]),
 "K20 Previous-tick warm start of the flow sampler":                                                     ([3,3,3,2,2,3,2,2,3], [3,3,3,2,2,3,2,2,3]),
 "K17 Edge-LLM action-runner + KV-reuse runtime":                                                        ([4,1,2,2,3,3,1,3,2], [4,1,2,2,3,3,1,3,2]),
}
rows = []
for k, (s1, s2) in cands.items():
    rows.append((sum(s2), sum(W[c] * x for c, x in zip(crit, s2)), sum(s1), sum(W[c] * x for c, x in zip(crit, s1)), k, s2))
print(f"  {'candidate':<104} {'F1 eq':>5} {'F1 wt':>6} {'FIN eq':>6} {'FIN wt':>6}  " + " ".join(f"{c:>5}" for c in crit))
for eq2, wt2, eq1, wt1, k, s2 in sorted(rows, key=lambda r: -r[0]):
    print(f"  {k:<104} {eq1:>5} {wt1:>6.1f} {eq2:>6} {wt2:>6.1f}  " + " ".join(f"{x:>5}" for x in s2))
print("  Changes F1 -> FINAL (red-team fixes 3, 4, 7, 8 accepted or partly accepted): Behaviour neutral (3) for measurement studies; K1 Gap 4->3 (2601.07393, B2D-Robust),")
print("  Small 5->3 (dynamic strata INSUFFICIENT on 441x3; sub-tick needs P-021), Win 4->3 (comparator, not baseline), Role 5->4 (no I2-perf), Cost 5->4 (KC-L-gated headline);")
print("  K2 Gap 4->3 (AlpaGym InferenceEngine already batches Alpamayo), Upstr 5->4 (a port of NVIDIA's engine; race), Cost 4->5 (KC-L-free, $38–137);")
print("  K3 Gap 4->3, Small 3->2 (no rental; device undecided), Behav 5->4 (engines need the ladder), Role 4->3 (no on-device line), Cost 3->2. Rejected: K1 AV 5->4 (criterion 1 is about the deployed stack, not the posting).")
print("  Merges: K14 -> K22 (P-021); K11, K4, K5, K15 -> K21 or K1's lever arms; K10, K7's device half, K8 (week-1 step) -> K3. Final order after merges: K22 > K21 > K3 under both weightings.")

print("\n" + "=" * 78)
print("F11. FOCUS-path scorecard (same nine criteria; FINAL)")
print("=" * 78)
paths = {"Path 1 research head (instrument composite with the conditional curve + lever arms)": [5,3,5,4,4,5,5,5,5],
         "Path 2 engineering on Jetson AGX Thor (as the measurement artifact; device-gated)": [5,3,2,3,4,4,3,3,2],
         "Path 3 shadow mode in HCMC (open-loop)": [3,4,1,2,3,3,2,2,1]}
for k, s in paths.items(): print(f"  {k:<84} equal {sum(s):>2}  wtd {sum(W[c]*x for c, x in zip(crit, s)):>5.1f}")

print("\n" + "=" * 78)
print("F12. Capstone cloud cost (DR1, 2026-10-28), redesigned pilot per F2 fix 2; ±2x on the s/step basis")
print("=" * 78)
e1_steps = 20 * 3 * 200 * 5            # smoke + positive control: d ∈ {0,1,2,3,10} ticks on 20 scenes
e2_steps = 3 * 60 * 3 * 200 * 2        # targeted pilot: 3 dynamic-actor strata x 60 scenes x 3 reps x {1,2} ticks
e3t_steps = 3 * 163 * 3 * 200 * 2      # targeted curve: 163 scenes/stratum = n for Δ=5 pp at σ_d 0.3, m=3 (F13)
e3w_steps = 3 * STEPS_SWEEP            # whole-set curve: 441 x 3 x {1,2,3} ticks
e4_steps = 3 * 60 * 3 * 200            # command-FIFO arm at d=2 on the same strata
e5_steps = 3 * 60 * 3 * 200            # sub-tick arm (139 vs 200 applied) on the same strata
def cost(steps):
    g = gpuh(steps); return g, (g[0] * P_WS[0], g[1] * P_WS[1])
items = {}
for name, st in [("E1 smoke + positive control (20 x 3 x {0,1,2,3,10})", e1_steps), ("E2 targeted pilot (3 strata x 60 x 3 x {1,2})", e2_steps),
                 ("E3t targeted curve (3 strata x 163 x 3 x {1,2})", e3t_steps), ("E3w whole-set curve (441 x 3 x {1,2,3})", e3w_steps),
                 ("E4 command-FIFO interface arm (3 x 60 x 3, d=2)", e4_steps), ("E5 sub-tick arm (3 x 60 x 3)", e5_steps)]:
    g, c = cost(st); items[name] = c
    print(f"    {name:<52} {st:>9,} steps -> {g[0]:5.1f}–{g[1]:5.1f} GPU-h -> ${c[0]:.0f}–${c[1]:.0f}")
fixed = {"B1 batch curve, timing-only": (4, 13), "B3 equivalence sweep (non-invariant arm)": (34, 124), "x86 SM120 dev": (10, 29)}
for k, v in fixed.items(): print(f"    {k:<52} ${v[0]}–${v[1]}")
sel = ["E1 smoke + positive control (20 x 3 x {0,1,2,3,10})", "E2 targeted pilot (3 strata x 60 x 3 x {1,2})", "E3t targeted curve (3 strata x 163 x 3 x {1,2})",
       "E4 command-FIFO interface arm (3 x 60 x 3, d=2)", "E5 sub-tick arm (3 x 60 x 3)"]
lo = sum(items[k][0] for k in sel) + sum(v[0] for v in fixed.values()); hi = sum(items[k][1] for k in sel) + sum(v[1] for v in fixed.values())
print(f"  total with the targeted curve (E3t), no B200: ${lo:.0f}–${hi:.0f}; with the optional B200 ISA check ($75–88): ${lo+75:.0f}–${hi+88:.0f}")
lo2 = lo - items[sel[2]][0] + items["E3w whole-set curve (441 x 3 x {1,2,3})"][0]; hi2 = hi - items[sel[2]][1] + items["E3w whole-set curve (441 x 3 x {1,2,3})"][1]
print(f"  total with the whole-set curve (E3w) instead: ${lo2:.0f}–${hi2:.0f}; the KC-L-free floor alone (B1 + B3 + E1 on synthetic inputs): ${4+34+items[sel[0]][0]:.0f}–${13+124+items[sel[0]][1]:.0f}")
print(f"  S5 node, 8 x H100, 24–48 h: vast.ai's one 8-GPU offer ${P_8XH100_VAST}/h -> ${P_8XH100_VAST*24:.0f}–${P_8XH100_VAST*48:.0f}; list ${P_8XH100_LIST[0]}–${P_8XH100_LIST[1]}/h -> ${P_8XH100_LIST[0]*24:.0f}–${P_8XH100_LIST[1]*48:.0f}")
print(f"  S4 training 225–450 H100-h at ${P_H100[0]}–${P_H100[1]}: ${225*P_H100[0]:.0f}–${450*P_H100[1]:.0f} (the plan's $403–1,080 used $1.79–2.40)")
print(f"  E3w concurrency if stacked on S3's 3–4 sweeps in one week: {(curve3[0]+3*sw[0])/168:.1f}–{(curve3[1]+4*sw[1])/168:.1f} GPUs continuously (F2 fix 9) -> spread over two weeks: {(curve3[0]+3*sw[0])/336:.1f}–{(curve3[1]+4*sw[1])/336:.1f}")

print("\n" + "=" * 78)
print("F13. Per-stratum power (ported from the red team's R1/R2/R4 with the same 03 §4 anchors; σ_d and ρ are ASSUMED inputs, not predictions)")
print("=" * 78)
def n_needed(delta_pp, sd, m=3, rho=0.6):
    n1 = (z * sd / (delta_pp / 100))**2; return n1, n1 * (1 + (m - 1) * rho) / m
print(f"  {'scenes':>7} | " + " | ".join(f"σ_d={sd:.1f}: MDE m=1 / m=3" for sd in (0.2, 0.3, 0.5)))
for n in (9, 15, 30, 60, 100, 163, 441):
    print(f"  {n:>7} | " + " | ".join(f"{mde(n,sd,1)[0]:5.1f} / {mde(n,sd,3)[0]:5.1f} pp" for sd in (0.2, 0.3, 0.5)))
print(f"  {'Δ (pp)':>7} | " + " | ".join(f"σ_d={sd:.1f}: n(m=1) / n(m=3)" for sd in (0.2, 0.3, 0.5)))
for d in (3, 5, 10, 20):
    print(f"  {d:>7} | " + " | ".join(f"{n_needed(d,sd)[0]:6.0f} / {n_needed(d,sd)[1]:6.0f}" for sd in (0.2, 0.3, 0.5)))
print("  -> a 9–30-scene stratum has an MDE of 8–36 pp at m=3: a sub-10 pp per-tick loss is INSUFFICIENT by construction on the 441 split.")
print("     Powering a dynamic-actor stratum for Δ = 5 pp needs 73–453 scenes (σ_d 0.2–0.5, m=3): only targeted sampling from the 2,202-scene curated pool can supply it [F2 row 30].")
print("     The E2 go/kill rule is therefore a pre-registered power criterion, not a point-estimate screen; a null without a registering positive control is 'not interpretable'.")

print("\n" + "=" * 78)
print("F14. Decode regime and the tick band: serial vs PR #45's Domino (8.30 accepted/round, 16 drafted) — the per-round cost is an ASSUMPTION")
print("=" * 78)
rounds = math.ceil(16 / PR45_ACCEPTED_PER_ROUND)
print(f"  serial: 16 tokens x {M_DECODE_MS_PER_TOK:.2f} ms = {16*M_DECODE_MS_PER_TOK:.0f} ms (Jetson proxy); Domino: {rounds} rounds for 16 tokens")
for per_round in (1.5, 2.5):
    tot = rounds * per_round * M_DECODE_MS_PER_TOK
    print(f"    if one verify round costs {per_round}x a serial token's weight read: {tot:.0f} ms for 16 tokens = {tot/16:.1f} ms/token effective (assumption)")
print(f"  expert step bound ≤ {ms(E_PARAMS*1 + kv_bf16, THOR_BW):.2f} ms (0.116e9) or {ms(E_PARAMS_LOW*1 + kv_bf16, THOR_BW):.2f} ms (0.050e9)")
print("  -> Under serial decode (what Edge-LLM ships for Alpamayo on Jetson today, FP16-only R1) n decides the band; under Domino (NVIDIA's DRIVE path, TRT 11.2 fork)")
print("     n and K move the band comparably and the student's expert-headroom question is reopened. Both regimes are carried; which one the student runs is a KC-E/T3 fact.")

print("\nDone.")
