# Arithmetic behind E_physloop_audit.md (inferences; inputs cited in the notes)
import datetime as dt, math
start = dt.date(2026, 11, 2)
print("12-week window:", start, "->", start + dt.timedelta(weeks=12))
# [C] effort per stage, 02 §3 table (v4.1 [C], hours)
stages = {"S1": (24, 24), "S2": (58, 64), "S3": (28, 34), "S4": (70, 78), "S5": (70, 78),
          "S6": (60, 62), "S7": (20, 24), "S8": (40, 40)}
tot = [sum(v[i] for v in stages.values()) for i in (0, 1)]
s2s7 = [sum(stages[s][i] for s in ["S2", "S3", "S4", "S5", "S6", "S7"]) for i in (0, 1)]
s5s7 = [sum(stages[s][i] for s in ["S5", "S6", "S7"]) for i in (0, 1)]
s6s7 = [sum(stages[s][i] for s in ["S6", "S7"]) for i in (0, 1)]
print("All [C] S1-S8:", tot, "| S2-S7:", s2s7, "| S5-S7:", s5s7, "| S6+S7:", s6s7)
for hw in (25, 30):
    cap = 12 * hw
    print(f"  capacity at {hw} h/wk = {cap} h; slack vs S2-S7 = {cap - s2s7[1]} to {cap - s2s7[0]} h")
print("Weeks for 370-405 h at 25-30 h/wk:", round(370/30, 1), "to", round(405/25, 1))

# 03 §4 S6 known answer: (100 - 49.5 - 7.3)/4.12 ; review §3.1 formula pieces
F, BW, eta = 517e12, 273e9, 0.4
for T in (2560, 2880):
    prefill = 2 * 2e9 * T / (eta * F) * 1e3
    expert = 4 * 0.5e9 / BW * 1e3          # K=4, 0.5B FP8 expert (decision-doc assumption)
    tok = 2e9 * 0.5625 / BW * 1e3           # 2B NVFP4 at 4.5 bits/param
    print(f"T={T}: prefill {prefill:.1f} ms, expert {expert:.2f} ms, decode {tok:.2f} ms/token, "
          f"tokens in 100 ms = {(100 - prefill - expert)/tok:.1f}")
# compact expert of PR #42 (28 layers x 384 wide, L5-7): ~12 d^2 per layer
d, L = 384, 28
p = 12 * d * d * L
print(f"PR #42 compact expert ~{p/1e6:.0f} M params (12*d^2*L estimate) vs 0.5B assumed")

# Thor-in-the-loop sweep: device time if one Jetson serves every drive() call serially
calls = 441 * 3 * 200
for ms in (100, 139, 171):
    print(f"  {calls} drive calls x {ms} ms = {calls*ms/1e3/3600:.1f} h of serial device time per sweep")

# Unreconciled scene counts (inference)
print("curated scenes 1761+441 =", 1761 + 441, "; PAI-AV 'alpasim' label =", 1536,
      "; difference =", 1761 + 441 - 1536)
