"""Track D arithmetic (2026-10-28). Every input is a sourced number; every output is an inference.
Run: python3 -I d_arith.py
Source IDs refer to D_hardware_reality.md section 7."""

T = 1e12

# ---------------- Device inputs (dense TFLOPS unless noted) ----------------
# Thor T5000: FP4 dense 1035 and FP8 dense 517 (NVIDIA Jetson Thor blog, ledger 10-03/10-25 C2) [DT2];
# BF16 dense = 517/2 (derived; NVIDIA prints none). BW 273 GB/s [DT1]. 2560 cores [DT1]; SMs 20 [DT9 user runtime].
# Max GPU clock 1.575 GHz MAXN, 1.386 GHz 120 W, 1.530 GHz 90/70 W with 6 of 10 TPCs [DT3].
thor = dict(fp4=1035, fp8=517, bf16=258.5, bw=0.273, sms=20, clk=1.575)
# RTX PRO 6000 WS: 4000 AI TOPS = FP4 sparse; FP32 126 TFLOPS; 24,064 cores; 1792 GB/s [DP1].
# Server: 1597 GB/s; tensor peaks as WS per Server page (ledger 10-03) [DP3]. Max-Q: 3511 AI TOPS sparse FP4, 110 TFLOPS FP32, 1792 GB/s [DP2].
ws = dict(fp4=4000 / 2, fp8=4000 / 4, bf16=4000 / 8, bw=1.792, cores=24064, fp32=126)
srv = dict(fp4=2000, fp8=1000, bf16=500, bw=1.597)
maxq = dict(fp4=3511 / 2, fp8=3511 / 4, bf16=3511 / 8, bw=1.792, cores=24064, fp32=110)
# H100 SXM: FP8 dense 1979, BF16 dense 989.5, 3.35 TB/s (ledger 10-03 Q2).
h100 = dict(fp4=None, fp8=1979, bf16=989.5, bw=3.35)
# HGX B200 (8 GPUs): FP4 dense 72 PF; FP8 sparse 72 -> dense 36; BF16 sparse 36 -> dense 18 [DP4]. BW 8 TB/s/GPU (ledger 10-03).
b200 = dict(fp4=72e3 / 8, fp8=36e3 / 8, bf16=18e3 / 8, bw=8.0)
# HGX B300: FP4 dense 108 PF; FP8 dense 36; BF16 dense 18 [DP4]. BW per GPU NOT on the HGX page (8 TB/s is secondary only).
b300 = dict(fp4=108e3 / 8, fp8=36e3 / 8, bf16=18e3 / 8, bw=8.0)

print('== Ridge points, FLOP/byte (dense peak / DRAM bandwidth)')
for name, d in [('Thor T5000 MAXN', thor), ('RTX PRO 6000 WS', ws), ('RTX PRO 6000 Server', srv),
                ('RTX PRO 6000 Max-Q', maxq), ('H100 SXM', h100), ('B200', b200), ('B300 (BW secondary)', b300)]:
    r = {k: (d[k] * T) / (d['bw'] * T) if d.get(k) else None for k in ('bf16', 'fp8', 'fp4')}
    print(f"  {name:22s} BF16 {r['bf16']:7.0f}  FP8 {r['fp8']:7.0f}  FP4 " + (f"{r['fp4']:7.0f}" if r['fp4'] else '    n/a'))

print('== Derived clocks and SM counts')
for name, d in [('RTX PRO 6000 WS', ws), ('RTX PRO 6000 Max-Q', maxq)]:
    clk = d['fp32'] * T / (2 * d['cores']) / 1e9
    print(f"  {name}: boost clock = FP32/(2*cores) = {clk:.3f} GHz; SMs = cores/128 = {d['cores'] / 128:.0f}")
ws_clk = ws['fp32'] * T / (2 * ws['cores']) / 1e9
print(f"  Thor: cores/128 = {2560 / 128:.0f} SMs; FP32 = 2560*2*1.575 GHz = {2560 * 2 * 1.575e9 / T:.3f} TF (staff: 8.064), "
      f"at 1.386 GHz = {2560 * 2 * 1.386e9 / T:.3f} TF (staff: 7.096)")

print('== Dense tensor FLOP per clock per SM')
for prec in ('fp8', 'fp4'):
    t = thor[prec] * T / (thor['sms'] * thor['clk'] * 1e9)
    r = ws[prec] * T / (188 * ws_clk * 1e9)
    print(f"  {prec.upper()}: Thor {t:,.0f}  RTX PRO 6000 WS {r:,.0f}  ratio {t / r:.1f}x")

print('== Thor peak by power mode (assumes tensor peak scales with SMs x clock; BW unknown below 4 FBPs)')
for mode, sms, clk in [('MAXN (10 TPC)', 20, 1.575), ('120 W default (10 TPC)', 20, 1.386), ('90 W / 70 W (6 TPC)', 12, 1.530)]:
    s = (sms / 20) * (clk / 1.575)
    print(f"  {mode:24s} scale {s:.3f}: FP4 {1035 * s:6.0f}  FP8 {517 * s:5.0f}  BF16 {258.5 * s:5.1f} TF;"
          f" FP8 ridge at 273 GB/s {517 * s / 0.273:5.0f}")
print(f"  If GPU DRAM bandwidth scaled with FBPs (3 of 4; UNVERIFIED): {273 * 3 / 4:.1f} GB/s")

print('== Measured vs peak on Thor')
for what, meas, peak in [('CUTLASS dense FP4 (forum, CUDA 13.0, 130 W)', 875, 1035), ('same, best', 878, 1035),
                         ('cuBLAS FP16 (forum, sizes unknown)', 130, 258.5), ('copy kernel GB/s (forum)', 200, 273)]:
    print(f"  {what:44s} {meas} / {peak} = {meas / peak:.1%}")

print('== Qwen3-8B NVFP4 decode bound on Thor (TRT-Edge-LLM table: 47.3 tok/s, batch 1, past KV 2048)')
L, H, I, kvh, hd, V = 36, 4096, 12288, 8, 128, 151936      # config.json [DQ1]
per_layer = H * H + 2 * H * kvh * hd + H * H + 3 * H * I  # q, k, v, o, gate, up, down
lin = L * per_layer
lm = V * H
nvfp4 = 4.5 / 8  # 4-bit values + one FP8 scale per 16 elements
kv_fp16 = L * kvh * hd * 2 * 2 * 2048
lo = lin * nvfp4 + lm * nvfp4 + kv_fp16 / 2
hi = lin * nvfp4 + lm * 2 + kv_fp16
print(f"  non-embedding linear params {lin / 1e9:.3f} B; lm_head {lm / 1e6:.1f} M; KV(FP16, 2048) {kv_fp16 / 1e9:.3f} GB")
for tag, b in [('lm_head NVFP4 + FP8 KV', lo), ('lm_head FP16 + FP16 KV', hi)]:
    bound = 273e9 / b
    print(f"  {tag:24s}: {b / 1e9:.3f} GB/token -> bound {bound:5.1f} tok/s -> measured 47.3 = {47.3 / bound:.0%} of bound")

print('== Emulating Thor on a proxy (FP8 dense; inference)')
rows = [
    ('Thor T5000 MAXN', 517, 273, 32 * 1.048576, 'tcgen05 (sm_110a)'),
    ('RTX PRO 6000 WS, green ctx 24 SMs', 1000 * 24 / 188, 1792, 128, 'mma.sync (sm_120a)'),
    ('RTX PRO 6000 WS, green ctx 16 SMs', 1000 * 16 / 188, 1792, 128, 'mma.sync (sm_120a)'),
    ('RTX PRO 6000 Server MIG 1g.24gb', 1000 / 4, 1597 / 4, 128 / 4, 'mma.sync (sm_120a)'),
    ('B200 green ctx 24 SMs (148 SMs, unverified)', 4500 * 24 / 148, 8000, 126, 'tcgen05 (sm_100a)'),
    ('B200 MIG 1g.23gb', 4500 / 7, 8000 / 8, 126 / 8, 'tcgen05 (sm_100a)'),
    ('H100 SXM MIG 1g.10gb', 1979 / 7, 3350 / 8, 50 / 8, 'wgmma (sm_90a)'),
]
print(f"  {'config':44s} {'FP8 TF':>7s} {'GB/s':>6s} {'L2 MB':>6s} {'ridge':>6s}  MMA")
for n, f, bw, l2, mma in rows:
    print(f"  {n:44s} {f:7.0f} {bw:6.0f} {l2:6.1f} {f * 1e12 / (bw * 1e9):6.0f}  {mma}")

print('== Working set vs L2 (L2: Thor 32 MiB user-reported; B200 126 MB; H100 50 MB; RTX PRO 6000 128 MB unverified)')
kv3k = 147456 * 3000  # bytes per token from 04 section 2
for n, l2 in [('Thor', 32 * 1.048576e6), ('H100', 50e6), ('B200', 126e6), ('RTX PRO 6000 (unverified 128 MB)', 128e6)]:
    print(f"  {n:32s} KV(3k tokens, BF16) {kv3k / 1e9:.3f} GB = {kv3k / l2:5.1f}x L2;  expert 5.0 GB/step = {5e9 / l2:6.0f}x L2")
print('  Weights in (33.6 MB, ~126 MB] can stay L2-resident across K flow steps on B200/RTX PRO 6000 but not on Thor:')
print('  re-read K times from DRAM on Thor vs ~once on the proxy (inference; K = 5-10).')

print('== Parallelism: CTAs for one M=64 skinny GEMM with 128-wide N tiles')
for n_out in (2048, 4096, 12288):
    ctas = n_out // 128
    print(f"  N={n_out:5d}: {ctas:3d} CTAs = {ctas / 188:5.1%} of 188 SMs, {min(ctas / 20, 1):5.1%} of Thor's 20 SMs (waves on Thor {ctas / 20:.1f})")

print("== Little's law, bytes in flight to saturate DRAM (latency is an ASSUMPTION)")
for n, bw, sms in [('Thor', 273e9, 20), ('RTX PRO 6000 WS', 1792e9, 188)]:
    for lat in (500e-9, 800e-9):
        inflight = bw * lat
        print(f"  {n:16s} lat {lat * 1e9:.0f} ns: {inflight / 1e6:.3f} MB in flight = {inflight / sms / 1e3:5.1f} KB/SM")

print('== Launch overhead illustration (kernel count from L7; per-launch cost is an ASSUMPTION)')
for c in (2e-6, 4e-6, 6e-6):
    print(f"  2,880 launches x {c * 1e6:.0f} us = {2880 * c * 1e3:.1f} ms per tick (budget 100 ms)")

print('== Prices (USD per GPU-hour)')
print('  Thor dev kit amortized (ASSUMPTION: 2,000 h use, MSRP 5,499, no resale): ', round(5499 / 2000, 2))
