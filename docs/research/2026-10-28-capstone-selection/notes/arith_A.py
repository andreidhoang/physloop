# Track A arithmetic (2026-10-28). Inputs are reported numbers; outputs are inferences.
BW = 273e9  # Jetson AGX Thor bytes/s (NVIDIA datasheet)
print("FlashDrive Table 1 (RTX PRO 6000), ms")
base, sysopt = 716.9, 513.3
rows = {"streaming": 355.8, "spec reasoning": 401.3, "adaptive steps": 450.4}
for k, v in rows.items(): print(f"  {k:15s} saves {sysopt - v:6.1f} ms vs sys-opt ({(sysopt - v) / sysopt:5.1%})")
print(f"  sum of individual savings {sum(sysopt - v for v in rows.values()):.1f}; 'All above' saves {sysopt - 176.0:.1f}")
print(f"  system opts save {base - sysopt:.1f} ms ({(base - sysopt) / base:.1%}); W4A8 saves {176.0 - 151.4:.1f} ms; Hz {1e3 / base:.2f} -> {1e3 / 151.4:.2f}")
print("FlashDrive Table 2 Jetson Thor: 1 sample", round(3770.3 / 943.6, 2), "x; 6 samples", round(14596.5 / 1522.6, 2), "x; 1-sample Hz", round(1e3 / 943.6, 2))
print("PR #45 DRIVE AGX Thor: FP8 Hz", round(1e3 / 171.2, 2), " FP16 Hz", round(1e3 / 220.6, 2), " FP8/FP16 speed-up", round(220.6 / 171.2, 3), " cut to 100 ms", round(171.2 - 100, 1), "ms =", f"{(171.2 - 100) / 171.2:.1%}")
print("Edge-LLM v0.10.0 Qwen3-VL-2B Jetson AGX Thor: NVFP4 decode ms/token", round(1e3 / 136.0, 2), " INT4AWQ", round(1e3 / 135.3, 2), " runtime NVFP4 BS1", round(1e3 / 151.9, 2))
print("  16 CoC tokens, no speculation, at 136 tok/s:", round(16 * 1e3 / 136.0, 1), "ms; prefill 2048 tok NVFP4 28.0 ms -> 3006 tok linear:", round(28.0 * 3006 / 2048, 1), "ms")
# Qwen3-VL-2B LM decode bound (assumptions: 28 layers, hidden 2048, 8 KV heads x 128, vocab 151936 tied, 1.41e9 non-embedding params at 4 bit, FP16 LM head, BF16/FP16 KV at 2048 past)
body = 1.41e9 * 0.5; head = 151936 * 2048 * 2; kv = 2048 * 28 * 8 * 128 * 2 * 2
t = (body + head + kv) / BW
print(f"  decode bound ~{t * 1e3:.2f} ms/token (body {body / 1e9:.2f} GB, LM head {head / 1e9:.2f} GB, KV {kv / 1e9:.3f} GB) -> measured/bound efficiency {t * 1e3 / (1e3 / 136.0):.0%}")
print("FlashRT PR #229 / docs, Jetson AGX Thor MAXN, Alpamayo 1.5-10B")
P_exp = 2.279e9; kv15 = 0.442e9
for name, ms10, bpp in [("balanced FP8 expert", 126.1, 1), ("bf16 expert", 213.0, 2)]:
    step = ms10 / 10; w = P_exp * bpp / BW * 1e3; wk = (P_exp * bpp + kv15) / BW * 1e3
    print(f"  {name}: {step:.2f} ms/step; bound weights-only {w:.2f}, +BF16 KV {wk:.2f} -> {w / step:.0%}-{wk / step:.0%} of bound")
P_vlm = 8.2e9
for name, ms, bpp in [("balanced W4A8 decode", 22.1, 0.5), ("bf16 decode", 67.4, 2)]:
    wk = (P_vlm * bpp + kv15) / BW * 1e3
    print(f"  {name}: {ms} ms/token; bound ~{wk:.1f} ms -> {wk / ms:.0%}")
print("  official 3322 ms -> Hz", round(1e3 / 3322, 2), "; fast 811 ->", round(1e3 / 811, 2), "; streaming best 345 ->", round(1e3 / 345, 2))
print("Edge-LLM release intervals (days): 07-23->08-12", 20, "; 08-12->09-03", 22, "; 09-03->09-29", 26, "; 09-29->10-28", 29)
