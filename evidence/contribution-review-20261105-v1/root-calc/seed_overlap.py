"""Known-answer illustration of AlpaSim main's additive seed derivation (affc2ea):
session_seed = random_seed + rollout_idx (engine.py L210-212);
inference_seed = session_seed + inference_count (driver main.py L713).
Compares with a hashed derivation (numpy SeedSequence). Pure arithmetic, no simulator."""
import numpy as np
S, reps, ticks = 1000, 3, 22
additive = {k: [S + k + i for i in range(ticks)] for k in range(reps)}
shared = set(additive[0]) & set(additive[1])
print("additive: replicate0 seeds", additive[0][:5], "... replicate1", additive[1][:5])
print("additive: seeds shared by replicate 0 and 1:", len(shared), "of", ticks)
print("additive: replicate1[t] == replicate0[t+1] for all t<21:",
      all(additive[1][t] == additive[0][t + 1] for t in range(ticks - 1)))
# two scenes with the same configured seed
print("two scenes, same random_seed, replicate 0: identical sequences:", additive[0] == [S + 0 + i for i in range(ticks)])
# zero seed
seed = 0; print("random_seed=0 -> session_seed:", (seed + 0) if seed else 0, "-> passed as", (0 or None))
def hashed(master, scene, rep, i):
    return int(np.random.SeedSequence([master, scene, rep, i]).generate_state(1)[0])
h = {(sc, k): [hashed(S, sc, k, i) for i in range(ticks)] for sc in range(441) for k in range(reps)}
allv = [v for seq in h.values() for v in seq]
print("hashed: total", len(allv), "unique", len(set(allv)))
