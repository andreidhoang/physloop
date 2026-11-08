"""Recompute D-022 / E19 oracle values and ratio examples (CPU, exact formulas).
Advantage per pinned Cosmos-RL grpo.py: (r - mean) / (np.std(r) + eps), eps=1e-6, population std."""
import numpy as np, math
eps = 1e-6
def adv(r):
    r = np.asarray(r, float); return (r - r.mean()) / (r.std() + eps)
print("G=2 gap 0.001:", adv([0, 0.001]))
print("G=2 tracking 0.1 m (-0.01*D):", adv([-0.213, -0.214]))
print("G=2 collision (1,-9):", adv([1, -9]))
print("G=2 gap 1e-4:", adv([0, 1e-4]), " gap 1e-5:", adv([0, 1e-5]))
print("G=6 one collision:", adv([1, 1, 1, 1, 1, -9]))
rng = np.random.default_rng(0)
noise = np.array([adv(rng.normal(0, 0.01, 6)) for _ in range(100000)])
print("G=6 noise-only rms:", np.sqrt((noise**2).mean()))
# geometric-mean vs joint ratio
lr = np.log([4.0, 1.0]); print("joint ratio:", math.exp(lr.sum()), " geo-mean ratio:", math.exp(lr.mean()))
# geo-mean 1.2 over 640 elements -> joint log ratio
print("joint log-ratio for geo 1.2 over 640 dims:", 640 * math.log(1.2))
# Gaussian KL, unit variance, mean diff 0.1
print("KL N(0,1)||N(0.1,1):", 0.5 * 0.1**2)
# progress_safety magnitudes: P in [0,1]? collision -10, offroad -5, D RMSE metres * -0.01
for D in (0.5, 2, 4, 10):
    print(f"distance term at RMSE {D} m:", -0.01 * D)
