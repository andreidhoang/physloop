# First-principles curriculum for an end-to-end autonomous-driving AI engineer: PhysLoop pillars P1–P7, verified 2026-10-03

*How these notes were made.* Identity checks used the arXiv API (39 IDs, all resolved), Crossref (journal and book records), arXiv HTML full text (formulas taken from MathML alttext), publisher, course and standards pages fetched over HTTPS, PubMed E-utilities, and two PDFs read page by page (PPO, RAND RR-1478). Every exercise answer was computed with numpy 2.5.1 on CPU, and the code is given inline. "Computed here" marks a derived number. A link marks a sourced fact. Access date for every row: **2026-10-03**. Nothing under `docs/` was edited.

## P1 — Driving as a decision problem: sources, derivations, exercise, trap

### Takeaway
Alpamayo's action space is a unicycle driven by (acceleration, curvature): 64 steps at 10 Hz over 6.4 s, giving 128 tokens. The paper's own Eq. 5 calls its update "Euler discretization", but it integrates heading exactly and position with the trapezoid rule. If you decode with plain forward Euler instead, you add about 0.6 m of endpoint error over 6.4 s. That is roughly 300× the error of the paper's own scheme.

### Cited Findings

#### Sources (ranked)
1. **Alpamayo-R1, §3.2.2, Eqs. 2–5.** The trajectory is 64 waypoints at 10 Hz over 6.4 s. It is represented through "unicycle dynamics with control inputs a = {(aⁱ, κⁱ)}" (acceleration, curvature), with ΔT = 0.1 s. Ground-truth controls are fit from logged waypoints by "least-squares formulation with Tikhonov regularization to attenuate high-frequency noise". There are 2 quantized values per waypoint, so 128 discrete tokens. — [arXiv 2511.00088 (HTML)](https://arxiv.org/html/2511.00088)
2. **Alpamayo 1.5 model card.** "the trajectory is represented as a sequence of dynamic actions (acceleration and curvature) following a unicycle model in bird's-eye-view (BEV) space". — [HF nvidia/Alpamayo-1.5-10B](https://huggingface.co/nvidia/Alpamayo-1.5-10B)
3. **Kong, Pfeiffer, Schildbach & Borrelli**, "Kinematic and dynamic vehicle models for autonomous driving control design", IEEE IV 2015, pp. 1094–1099, DOI 10.1109/IVS.2015.7225830. The record is verified; the model equations were *not* re-read (the PDF host did not resolve). — [Crossref/DOI](https://doi.org/10.1109/IVS.2015.7225830)
4. **Rajamani**, *Vehicle Dynamics and Control*, **2nd edition** (Springer, Mechanical Engineering Series, © 2012; hardcover published 21 Dec 2011, ISBN 978-1-4614-1432-2). Springer lists it as "Latest edition". — [Springer](https://link.springer.com/book/10.1007/978-1-4614-1433-9)
5. **Werling, Ziegler, Kammel & Thrun**, "Optimal trajectory generation for dynamic street scenarios in a Frenét Frame", ICRA 2010, pp. 987–993. The record is verified; the text was not re-read. — [DOI](https://doi.org/10.1109/ROBOT.2010.5509799)
6. **LaValle**, *Planning Algorithms*, Cambridge University Press 2006, 842 pp., ISBN 9780521862059. The whole book is free in HTML. — [lavalle.pl/planning](https://lavalle.pl/planning/); [Crossref DOI 10.1017/CBO9780511546877](https://doi.org/10.1017/CBO9780511546877)
7. **Thrun, Burgard & Fox**, *Probabilistic Robotics*, MIT Press 2005, 647 pp. Confirmed via a published review record; the MIT Press page returned 403. — [Artificial Life 14(2):227–229](https://doi.org/10.1162/artl.2008.14.2.227)
8. **MPC practice:** CMU 16-745 Optimal Control (Z. Manchester). The schedule includes "Convex Model-Predictive Control", "LQR in 3 Ways", "Differential Dynamic Programming & iLQR" and "Direct Trajectory Optimization, Collocation, & SQP". — [optimalcontrol.ri.cmu.edu](https://optimalcontrol.ri.cmu.edu/)
9. **Metrics.**
   - NAVSIM (Dauner et al., 2024) — [arXiv 2406.15349](https://arxiv.org/abs/2406.15349)
   - EPDMS. The pseudo-simulation paper says it "combines multiplicative penalties for rule violations with a weighted average of several subscores": a product of penalty filters times a weighted sum of subscore filters. The normalization of the weighted term was cut off in the extracted text and is not reproduced here. The metric builds on the PDMS. — [arXiv 2506.04218 (HTML)](https://arxiv.org/html/2506.04218)
   - Open-loop vs closed-loop. Bosch 2026 reports Spearman ρ = −0.36 (p = 0.43) between L2 and Bench2Drive Driving Score, "indicating no significant correlation". The NAVSIM PDMS gives ρ = 0.90 (p = 0.002) on n = 8 paired methods, and Ego Progress alone gives ρ = 0.83. — [arXiv 2605.00066 (HTML)](https://arxiv.org/html/2605.00066)
   - Dauner et al. 2023, "Parting with Misconceptions…" — [arXiv 2306.07962](https://arxiv.org/abs/2306.07962)

#### Derivations to own (from memory)
1. **Unicycle in (a, κ)** (exact-text anchor: [AR1 Eq. 5](https://arxiv.org/html/2511.00088)):
   - θ⁽ⁱ⁺¹⁾ = θⁱ + ΔT κⁱ vⁱ + (ΔT²/2) κⁱ aⁱ
   - v⁽ⁱ⁺¹⁾ = vⁱ + ΔT aⁱ
   - x⁽ⁱ⁺¹⁾ = xⁱ + (ΔT/2)(vⁱ cos θⁱ + v⁽ⁱ⁺¹⁾ cos θ⁽ⁱ⁺¹⁾); y uses sin in the same way.

   Derive the θ line yourself. Integrate θ̇ = κ v(t) with v(t) = vⁱ + aⁱt over one step: Δθ = κ(vⁱΔT + aⁱΔT²/2). So the heading update is *exact* for piecewise-constant (a, κ), and only position carries discretization error (computed here).
2. **Why (a, κ) and not raw waypoints.** The paper's stated reason: "controllers typically smooth trajectory outputs to ensure consistent and stable execution on-vehicle. Thus, instead of directly learning τ in the raw position waypoint space, we adopt an action representation governed by unicycle dynamics that leads to better closed-loop performance." — [AR1](https://arxiv.org/html/2511.00088). First-principles reasons are listed under Inferences.
3. **Local truncation error and order.** Forward Euler has O(ΔT) global error and the trapezoid rule O(ΔT²). This is demonstrated numerically below: halving ΔT halves the Euler error and quarters the Eq. 5 error (computed here).
4. **Open-loop vs closed-loop metric structure.**
   - minADEₖ: the minimum over k samples of the mean L2 error over the horizon (standard definition; not re-read from a source here).
   - EPDMS: a product of penalty gates times a weighted average of subscores ([2506.04218](https://arxiv.org/html/2506.04218)).
   - Why they rank differently: in the Bosch study, safety-maximizing methods "rank highly in NAVSIM but underperform in closed-loop due to timeout and slow-driving penalties" ([2605.00066 abstract](https://arxiv.org/abs/2605.00066)).

#### Exercise (known answer, no GPU)
Drive a constant-speed arc: v = 10 m/s, κ = 0.02 m⁻¹ (R = 50 m), T = 6.4 s, and compare against the closed form x = sin(vκT)/κ, y = (1 − cos vκT)/κ.

**Answer (computed here):**
- Exact endpoint is (47.9008, 35.6642) m, with θ = 1.28 rad.
- Forward Euler at ΔT = 0.1: endpoint error **0.5972 m**. At ΔT = 0.05: 0.2986 m.
- AR1 Eq. 5 at ΔT = 0.1: **0.00199 m**. At ΔT = 0.05: 0.00050 m.

```python
import math
def exact(v,k,T): th=v*k*T; return math.sin(th)/k,(1-math.cos(th))/k
def fwd_euler(v,k,dt,n):
    x=y=th=0.0
    for _ in range(n): x+=dt*v*math.cos(th); y+=dt*v*math.sin(th); th+=dt*v*k
    return x,y
def ar1_eq5(v,a,k,dt,n):          # AR1 Eq.5: exact heading, trapezoid position
    x=y=th=0.0
    for _ in range(n):
        v2=v+dt*a; th2=th+dt*k*v+0.5*dt*dt*k*a
        x+=0.5*dt*(v*math.cos(th)+v2*math.cos(th2)); y+=0.5*dt*(v*math.sin(th)+v2*math.sin(th2)); v,th=v2,th2
    return x,y
xe,ye=exact(10,0.02,6.4)
for dt in (0.1,0.05):
    n=round(6.4/dt); print(dt, math.dist(fwd_euler(10,.02,dt,n),(xe,ye)), math.dist(ar1_eq5(10,0,.02,dt,n),(xe,ye)))
```

**How to check:** compare against the closed-form arc. The error ratio when ΔT is halved reveals the order (2× means first order; 4× means second order).

#### The trap
- **Taking "Euler discretization" literally.** Implementing the decoder with forward Euler because the paper uses that word gives a 0.60 m systematic endpoint error at 6.4 s. That is the same order as the 0.913 m minADE6 headline that docs/03 §2 sets as nanoAlpamayo's acceptance target (computed here; the 0.913 m figure is from docs/03 and is not verified in this note).
- **Reading Bosch's "ρ = −0.36" as anti-correlation.** It is ρ = −0.36 with p = 0.43 on n = 8, which the paper itself calls "no significant correlation" — [2605.00066](https://arxiv.org/html/2605.00066).

### Inferences
- First-principles reasons for (a, κ), beyond the paper's controller-smoothness argument:
  - Any (a, κ) sequence integrates to a path that the kinematic model can drive, so feasibility holds by construction.
  - κ is geometry that does not depend on speed, which separates lateral from longitudinal control.
  - Bounded physical ranges make uniform binning into 2 × 64 tokens natural.
- Mapping κ to a steering angle needs a bicycle model, for which Kong et al. 2015 is the canonical source.
- docs/03 P1's derivation target is fully consistent with AR1: 64 waypoints, 10 Hz, 6.4 s, (accel, curvature).
- The Stage-0 notebook should implement Eq. 5 exactly, not textbook Euler.

### Gaps
- Kong et al. 2015 and Werling et al. 2010: only the records were verified, not the text. Their equations should be checked against the PDFs before being quoted:
  - Kong's kinematic bicycle (slip angle at the CG, curvature–steering relation).
  - Werling's Frenet decomposition and jerk-optimal quintic polynomials.

  The quintic-from-minimum-jerk result can be derived by calculus of variations. Whether Werling uses exactly that cost was not confirmed here.
- LaValle and Thrun et al.: chapter-level contents (sampling-based planners, Bayes filters) were not re-read. Only identity and the free-access status were confirmed.

## P2 — VLA modeling: sources, derivations, exercise, trap

### Takeaway
docs/03's "180 tokens/image at 320×576" figure is consistent only with a **Qwen3-VL-class** encoder: 16-px patches with a 2×2 merge, i.e. 32 px per token. Cosmos-Reason2 (the Alpamayo 1.5 backbone) is based on Qwen3-VL. The AR1 paper itself describes a 28-px-per-token scheme, which would give 231 tokens. A π0-style flow action expert reuses the cached prefix keys and values and recomputes only the action suffix at each of its 10 Euler steps.

### Cited Findings

#### Sources (ranked)
1. **Vaswani et al. 2017.** Attention(Q,K,V) = softmax(QKᵀ/√d_k)V. "We compute the dot products of the query with all keys, divide each by √d_k". — [arXiv 1706.03762 (HTML)](https://arxiv.org/html/1706.03762)
2. **Dosovitskiy et al. 2020 (ViT).** The image is reshaped into "a sequence of flattened 2D patches x_p ∈ ℝ^{N×(P²·C)}" with "N = HW/P²". — [arXiv 2010.11929 (HTML)](https://arxiv.org/html/2010.11929)
3. **Qwen2.5-VL technical report.**
   - Images are "resized to multiples of 28"; patches have "a stride of 14"; the encoder groups "spatially adjacent sets of four patch features", concatenated and passed through "a two-layer multi-layer perceptron (MLP)". The configuration table lists patch size 14. — [arXiv 2502.13923 (HTML)](https://arxiv.org/html/2502.13923)
   - Official config: patch_size 14, spatial_merge_size 2, temporal_patch_size 2. — [HF Qwen2.5-VL-7B config](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/raw/main/config.json)
   - The resize rule in code is `h_bar = round(height / factor) * factor`, with factor = 28. — [transformers qwen2_vl image processor](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py)
4. **Qwen3-VL technical report.**
   - "As in Qwen2.5-VL, we use a two-layer MLP to compress 2×2 visual features from the vision encoder into a single visual token"; the default encoder is SigLIP2-SO-400M. — [arXiv 2511.21631 (HTML)](https://arxiv.org/html/2511.21631)
   - Official config: patch_size **16**, spatial_merge_size 2, temporal_patch_size 2. Text config: 36 layers, 32 attention heads, **8 KV heads**, head_dim 128. — [HF Qwen3-VL-8B config](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/raw/main/config.json)
   - "Cosmos-Reason2 is based on the Qwen3-VL architecture." — [nvidia-cosmos/cosmos-reason2 README](https://raw.githubusercontent.com/nvidia-cosmos/cosmos-reason2/main/README.md)
   - Alpamayo 1.5 lists base_model nvidia/Cosmos-Reason2-8B, with an 8.2B backbone and a 2.3B action expert. — [HF model card](https://huggingface.co/nvidia/Alpamayo-1.5-10B)
5. **LLaVA (Liu et al. 2023).** "we apply a trainable projection matrix W to convert Z_v into language embedding tokens H_v". Stage 1 trains "θ = W (the projection matrix) only". — [arXiv 2304.08485 (HTML)](https://arxiv.org/html/2304.08485)
6. **π0 (Black et al. 2024; v4 2026-01-08).**
   - A separate "action expert … produces continuous actions via flow matching".
   - Action chunk H = 50; forward Euler with "10 integration steps (corresponding to δ = 0.1)".
   - "inference can be implemented efficiently by caching the attention keys and values for the prefix o_t and only recomputing the suffix".
   - A "blockwise causal attention mask with 3 blocks".
   - Timestep p(τ) = Beta((s−τ)/s; 1.5, 1) with s = 0.999.

   — [arXiv 2410.24164 (HTML)](https://arxiv.org/html/2410.24164)
7. **FAST (Pertsch et al. 2025).** "Given a normalized chunk of actions, we apply discrete cosine transform (DCT)… quantize the DCT coefficients and use byte-pair encoding (BPE) to compress the flattened sequence". The motivation is that high-frequency actions produce "strong correlation between consecutive tokens". — [arXiv 2501.09747 (HTML)](https://arxiv.org/html/2501.09747)
8. **AR1 tokenization and decoder.**
   - Example: W×H image → W/14 × H/14 patch features, "2× bilinearly downsampled" to W/28 × H/28, so "W = 448, H = 280 … produces 160 tokens per image".
   - The paper's "Multi-Camera Tokenization" variant (p_x = p_y = p_z = 8) needs "only 288 tokens … to represent one timestep of observations, irrespective of the number of cameras or their resolution": about 41.1 tokens per image for 7 cameras, 3.9× fewer than single-image tokenization.
   - Flow decoder target u(a_t | a) = a − ε, with Euler integration δ_t = 0.1.

   — [arXiv 2511.00088 (HTML)](https://arxiv.org/html/2511.00088)

#### Derivations to own
1. **√d_k scaling.** If q and k have i.i.d. unit-variance components, q·k has variance d_k. Dividing by √d_k keeps the softmax logits O(1) (math; formula anchored to [Vaswani](https://arxiv.org/html/1706.03762)).
2. **Visual token count.** N = (H′/(P·m)) · (W′/(P·m)), where H′, W′ are the sizes after resizing to multiples of P·m. That gives 32 px per token for Qwen3-VL and 28 px per token for Qwen2.5-VL ([ViT](https://arxiv.org/html/2010.11929), [configs](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/raw/main/config.json)).
3. **KV-cache bytes.** 2 (K and V) × L × n_kv × d_head × bytes per element × tokens (math). GQA divides this by n_heads / n_kv.
4. **Prefix/suffix split in a flow action expert.** The VLM prefix KV is computed once; the action-token suffix is recomputed at each of the N Euler steps ([π0](https://arxiv.org/html/2410.24164)). Expert cost therefore scales with N × (suffix work), plus reading the cached prefix each step.

#### Exercise (known answer, no GPU)
How many visual tokens and how much BF16 KV cache do 4 cameras × 4 frames at 320×576 produce on a Qwen3-VL-8B-class backbone?

**Answer (computed here):**
- 320/32 × 576/32 = 10 × 18 = **180 tokens/image**.
- × 16 images = **2,880 tokens**.
- KV per token = 2 × 36 × 8 × 128 × 2 B = **147,456 B**, so **424.7 MB** for 2,880 tokens.

```python
def tokens(h,w,patch,merge=2): f=patch*merge; return round(h/f)*round(w/f)   # smart_resize rounds to nearest
print(tokens(320,576,16), tokens(320,576,14))    # 180 (Qwen3-VL), 231 (Qwen2.5-VL: 308x588)
kv=2*36*8*128*2; print(kv, kv*2880/1e6)           # 147456 B/token, 424.7 MB
```

**How to check:** reproduce the 180 and 2,880 figures from docs/03 P1/P2. Then re-run with patch = 14 and confirm 231.

#### The trap
- **Using the 14-px rule** (Qwen2.5-VL, or AR1's 28-px example) gives **231 tokens/image → 3,696 tokens**. With a ceil-style resize it gives 252 → 4,032.
- **Computing KV with all 32 attention heads** instead of the 8 KV heads gives **1.70 GB**, a 4× overestimate.
- **Assuming each frame is a separate image** when frames actually enter as one video with temporal_patch_size = 2. Four frames then fuse into 2 temporal groups, giving 4 × 2 × 180 = **1,440 tokens**, half the doc's figure (computed here). Which path Alpamayo 1.5 uses is not verified.

### Inferences
- The doc's 2,880 figure implicitly assumes per-frame image tokenization on a 32-px-per-token encoder. The video-mode alternative halves it. Settle this from the alpamayo1.5 code before using 2,880 in any latency budget.
- The "flow expert as a black box" trap (docs/03 P2) is best broken by deriving the prefix/suffix split. It explains why the expert is memory-bound: at batch 1 it re-reads weights and cached KV on every step.

### Gaps
- The Cosmos-Reason2-8B config and model card returned HTTP 401 (gated). Using Qwen3-VL-8B's text config (36 layers, 8 KV heads, head_dim 128) for Cosmos-Reason2 is an *assumption* based on the README's "based on the Qwen3-VL architecture".
- Qwen3-VL's resize factor (32) is inferred from patch 16 × merge 2. Its processor code was not read.

## P3 — Learning algorithms: sources, derivations, exercise, trap

### Takeaway
Six derivations were verified against the primary texts:
1. BC's T²ε versus DAgger's uTε.
2. The conditional-OT flow-matching target.
3. Flow-GRPO's ODE→SDE drift correction.
4. GRPO's group-normalized advantage and its k3 KL estimator.
5. PPO's clip.
6. Hinton's T² gradient scaling.

The best cheap exercise is a 1-D Gaussian flow with an exactly known velocity field. One Euler step collapses every sample to the mean; 10 steps leave about 1.9% relative MSE.

### Cited Findings

#### Sources (ranked)
1. **Ross & Bagnell 2010 — the BC compounding bound.** "a classifier that makes a mistake with probability ε under the distribution of states/observations encountered by the expert can make as many as T²ε mistakes in expectation". Restated as Theorem 2.1: J(π) ≤ J(π*) + T²ε, and called "tight" ([DAgger paper text](https://arxiv.org/html/1011.0686)). Venue: "Efficient Reductions for Imitation Learning", AISTATS 2010, PMLR 9:661–668 — [PMLR](https://proceedings.mlr.press/v9/ross10a.html)
2. **Ross, Gordon & Bagnell 2011 (DAgger).**
   - Theorem 3.1: if N is Õ(T), some policy in the sequence has E_{d_π̂}[ℓ] ≤ ε_N + O(1/T).
   - Bounded-Q lemma: J(π) ≤ J(π*) + uTε.

   — [arXiv 1011.0686 (HTML)](https://arxiv.org/html/1011.0686); AISTATS 2011, PMLR 15:627–635 — [PMLR](https://proceedings.mlr.press/v15/ross11a.html)
3. **Lipman et al. 2022 (Flow Matching).** OT conditional path μ_t(x) = t x₁, σ_t(x) = 1 − (1 − σ_min)t (Eq. 20), generated by u_t(x | x₁) = (x₁ − (1 − σ_min)x)/(1 − (1 − σ_min)t) (Eq. 21). The authors note that "the OT path's conditional vector field has constant direction in time". — [arXiv 2210.02747 (HTML)](https://arxiv.org/html/2210.02747)
4. **Lipman et al. 2024, *Flow Matching Guide and Code*.** "p_{t|1}(x|x₁) = N(x | t x₁, (1−t)² I) … also known as the conditional optimal-transport or linear path". Simplest loss: L = E‖u^θ_t(X_t) − (X₁ − X₀)‖², t ~ U[0,1]. — [arXiv 2412.06264](https://arxiv.org/abs/2412.06264) (text via [ar5iv](https://ar5iv.labs.arxiv.org/html/2412.06264))
5. **Liu, Gong & Liu 2022 (rectified flow).** "Euler method with step size 1/N"; the reflow procedure gives "nearly straight flows that yield good results even with the coarsest time discretization". — [arXiv 2209.03003 (HTML)](https://arxiv.org/html/2209.03003)
6. **Schulman et al. 2017 (PPO), Eq. 7.** L^CLIP(θ) = Ê_t[min(r_t(θ)Â_t, clip(r_t(θ), 1−ε, 1+ε)Â_t)], with r_t = π_θ/π_θold and "say, ε = 0.2". The final objective is "a lower bound (i.e., a pessimistic bound) on the unclipped objective". — [arXiv 1707.06347 PDF p. 3](https://arxiv.org/pdf/1707.06347)
7. **DeepSeekMath (Shao et al. 2024) — GRPO.**
   - Rewards "normalized by subtracting the group average and dividing by the group standard deviation": Â_{i,t} = (r_i − mean(r))/std(r).
   - KL by "the following unbiased estimator (Schulman, 2020)": D_KL[π_θ‖π_ref] = π_ref/π_θ − log(π_ref/π_θ) − 1, "guaranteed to be positive".

   — [arXiv 2402.03300 (HTML)](https://arxiv.org/html/2402.03300)
8. **Schulman, "Approximating KL Divergence" (blog).**
   - For samples x ~ q with r = p/q: k1 = −log r, k2 = ½(log r)², k3 = (r − 1) − log r.
   - Table for q = N(0,1), p = N(0.1,1) (true KL 0.005), bias/true and stdev/true: k1 0 / 20, k2 0.002 / 1.42, k3 0 / 1.42.
   - Table for p = N(1,1) (true KL 0.5): k1 0 / 2, k2 0.25 / 1.73, k3 0 / 1.7.

   — [joschu.net/blog/kl-approx.html](http://joschu.net/blog/kl-approx.html)
9. **Hinton, Vinyals & Dean 2015.** "Since the magnitudes of the gradients produced by the soft targets scale as 1/T² it is important to multiply them by T²". — [arXiv 1503.02531 (HTML)](https://arxiv.org/html/1503.02531)
10. **Agarwal et al. (GKD).** arXiv title "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes". "GKD trains the student on its self-generated sequences that are on-policy… using teacher probabilities as expert labels". It frames KD as "imitation learning problem with an interactive expert (Ross et al., 2011)". — [arXiv 2306.13649 (HTML)](https://arxiv.org/html/2306.13649)
11. **Flow-GRPO (Liu et al. 2025).**
    - The marginal-preserving reverse-time SDE dx = (v − (σ²/2)∇log p)dt + σ dw (Eq. 7), which for rectified flow becomes dx = [v + σ²/(2t)(x + (1−t)v)]dt + σ dw (Eq. 8).
    - The Euler–Maruyama step adds σ_t√Δt ε (Eq. 9), with σ_t = a√(t/(1−t)).
    - Each denoising step is therefore an isotropic Gaussian policy, so log-probabilities are tractable.

    — [arXiv 2505.05470 (HTML)](https://arxiv.org/html/2505.05470)
12. **π_RL (Chen et al. 2025).** "(1) Flow-Noise models the denoising process as a discrete-time MDP with a learnable noise network for exact log-likelihood computation. (2) Flow-SDE … formulating a two-layer MDP that employs ODE-to-SDE conversion". The noise network "is trained jointly with the velocity but discarded after fine-tuning". — [arXiv 2510.25889 (HTML)](https://arxiv.org/html/2510.25889)
13. **Driving-specific post-training.**
    - OPTED: "a privileged teacher is trained using RL on vectorized inputs (HD-map and bounding boxes)" and then supervises the camera student in AlpaSim; it uses "approximately three orders of magnitude fewer simulator interactions than direct RL post-training". — [arXiv 2609.20756](https://arxiv.org/abs/2609.20756)
    - RoaD: closed-loop *supervised* fine-tuning on the policy's own expert-guided rollouts; in AlpaSim it "improves driving score by 41% and reduces collisions by 54%". — [arXiv 2512.01993](https://arxiv.org/abs/2512.01993)

#### Derivations to own
1. **T²ε versus uTε** ([1011.0686](https://arxiv.org/html/1011.0686)). Telescoping over the policies π_{1:t} (run the learner for t steps, then the expert) gives J(π) = J(π*) + Σ_t E[Q*(s,π) − Q*(s,π*)] ≤ J(π*) + uTε.
2. **The conditional-OT target and the marginal field.** With X_t = (1−t)X₀ + tX₁, the CFM regression target is X₁ − X₀. Its minimizer is u_t(x) = E[X₁ − X₀ | X_t = x]. For a Gaussian X₁ ~ N(μ, s²) this is u_t(x) = μ + β(t)(x − tμ), with β(t) = (ts² − (1−t))/((1−t)² + t²s²) (math, checked by regression below).
3. **ODE → SDE.** For the rectified-flow parametrization, ∇log p_t(x) = −(x + (1−t)v)/t. Substituting into v − (σ²/2)∇log p reproduces Flow-GRPO Eq. 8 exactly (derived here; consistent with [Eq. 7→8](https://arxiv.org/html/2505.05470)).
4. **GRPO/PPO mechanics.**
   - Group-normalized advantages.
   - k3 = (r − 1) − log r is unbiased and always ≥ 0, with r = π_ref/π_θ and samples drawn from π_θ.
   - Clipping removes the gradient whenever r leaves [1−ε, 1+ε] in the direction that would improve the objective ([PPO](https://arxiv.org/pdf/1707.06347), [DeepSeekMath](https://arxiv.org/html/2402.03300)).

#### Exercise (known answer, no GPU)
Take X₀ ~ N(0,1) and X₁ ~ N(μ = 2, s = 0.5) on the linear path. The exact transport map is x₁ = μ + s·x₀. Each Euler step multiplies the deviation (x − tμ) by (1 + Δt β(tₖ)).

**Answer (computed here, exact product):**
- **1 step:** output std **0** (every sample collapses to μ = 2), relative MSE 100%.
- 2 steps: std 0.200 (36%).
- 4 steps: 0.3405 (10.2%).
- **10 steps:** std **0.43078**, relative MSE **1.92%**.
- 20 steps: 0.464 (0.51%).
- 100 steps: 0.49265 (0.02%).

A least-squares regression of (X₁ − X₀) on X_t at t = 0.3 recovers slope −1.2234 and intercept 2.7353, against analytic values −1.2195 and 2.7317.

```python
mu,s=2.0,0.5
beta=lambda t:(t*s*s-(1-t))/((1-t)**2+(t*s)**2)      # exact marginal field u_t(x)=mu+beta(t)(x-t mu)
for N in (1,2,10,100):
    g=1.0
    for k in range(N): g*=1+beta(k/N)/N              # Euler scales the deviation (x - t mu)
    print(N, g, (g-s)**2/s**2)                        # output std, relative MSE vs exact map
```

**How to check:**
- The mean must be exactly μ at every N.
- The std must converge to s = 0.5 as N grows.
- A tiny CPU MLP trained with the CFM loss should match β(t) at any t.

**Secondary quick checks:**
- KL estimators (2M samples, seed 0), reproducing Schulman's table: at μ = 0.1, k1 stdev/true 19.99, k2 1.417, k3 1.416. At μ = 1, k2 bias/true 0.250. Analytically, k2's bias/true is exactly μ²/4 (0.0025 at μ = 0.1; the blog prints 0.002).
- GRPO with rewards [1,0,0,0]: advantages [1.732, −0.577, −0.577, −0.577] with population std, or [1.5, −0.5, −0.5, −0.5] with sample std.
- PPO terms at ε = 0.2: (r=1.3, A=+1) → 1.2; (0.7, +1) → 0.7; (0.7, −1) → −0.8; (1.3, −1) → −1.3.
- DAgger vs BC at T = 64, ε = 0.01: uTε = 0.64u versus T²ε = 40.96. The BC bound is nearly vacuous against a maximum cost of 64.

#### The trap
- **"Conditional OT paths are straight, so a few Euler steps suffice."** The *marginal* trajectories under independent coupling are curved. One step maps everything to the mean, and 10 steps still shrink the spread by 13.8% (0.431 vs 0.5).
- **Inverting the k3 ratio** (r = π_θ/π_ref) gives E = 1.218 against a true KL of 0.5 at μ = 1, i.e. bias/true +1.44 (computed here).
- **The GRPO std convention.** The paper writes std(r) without specifying population or sample std, which changes advantages by about 15% on a group of 4.

### Inferences
- docs/03 §4 Stage 0 sets the acceptance "toy flow model within 2% MSE of the analytic OT path". In this toy, the Euler sampler alone uses 1.92% of that budget at 10 steps. The criterion should name:
  - the coupling;
  - the step count;
  - whether "OT path" means the *conditional* straight path or the *marginal* map.
- The marginal map μ + s·x₀ is the 1-D OT map, but its trajectory is not a straight line. So the phrase "analytic OT path" is ambiguous as written.
- GRPO's k3 estimator is exactly Schulman's k3 with q = π_θ and p = π_ref. The inverted-ratio bug yields a quantity that is not a KL divergence.

### Gaps
- Flow-GRPO's σ_t = a√(t/(1−t)) diverges at t → 1. How implementations avoid t = 1 was not checked.
- The exact form of AlpaGym's Flow-SDE was not checked (outside this scope).

## P4 — Data engine: sources, derivations, exercise, trap

### Takeaway
The core math to own:
- NeRF's volume-rendering quadrature, i.e. alpha compositing.
- 3DGS's EWA covariance projection.
- A one-line leakage calculation. A random *frame* split with 20 frames per location puts about 99% of locations in both train and validation.

EUVS documents that current novel-view synthesis methods overfit to training views under extrapolation.

### Cited Findings

#### Sources (ranked)
1. **Kerbl et al. 2023 (3DGS)**, ACM TOG 42(4).
   - "Given a viewing transformation W the covariance matrix Σ′ in camera coordinates is given as follows: Σ′ = J W Σ Wᵀ Jᵀ, where J is the Jacobian of the affine approximation of the projective transformation" (Eq. 5, after Zwicker et al. 2001).
   - Color uses spherical harmonics "following standard practice".

   — [arXiv 2308.04079 (HTML)](https://arxiv.org/html/2308.04079)
2. **Mildenhall et al. 2020 (NeRF).** Ĉ(r) = Σᵢ Tᵢ(1 − exp(−σᵢδᵢ))cᵢ, with Tᵢ = exp(−Σ_{j<i} σⱼδⱼ). This "reduces to traditional alpha compositing with alpha values αᵢ = 1 − exp(−σᵢδᵢ)" (Eq. 3). — [arXiv 2003.08934 (HTML)](https://arxiv.org/html/2003.08934)
3. **EUVS (Han et al., ICCV 2025).**
   - Three settings: translation only (lane change), rotation only, translation + rotation.
   - In the Figure-1 example, the drops relative to training are "28% for PSNR, 22% for SSIM, and 50% for LPIPS".
   - Conclusion: "incorporating diffusion priors and improving geometry cannot fundamentally improve NVS under large view changes".

   — [arXiv 2412.05256 (HTML)](https://arxiv.org/html/2412.05256); venue on the [project page](https://ai4ce.github.io/EUVS-Benchmark/)
4. **Instant NuRec (NVIDIA 2026).** A feed-forward model that turns "a short multi-view driving log into a fully simulatable 3DGS world in a single forward pass". It reconstructs a 10–20 s multi-camera scene "in roughly 1.5 seconds", scores +2.01 dB PSNR over the strongest baseline on Waymo Open, uses 3DGUT, and "is compatible with AlpaSim". — [arXiv 2607.14203](https://arxiv.org/abs/2607.14203)
5. **OmniDreams (NVIDIA 2026).** A generative world model mid- and post-trained from Cosmos on "21k hours of driving scenarios", deployed closed-loop "with the Alpamayo 1 policy model and AlpaSim orchestrator". — [arXiv 2606.03159](https://arxiv.org/abs/2606.03159)

#### Derivations to own
1. **Alpha compositing from the transmittance ODE.** dT/dt = −σT gives exp(−∫σ), and discretizing gives Tᵢ and αᵢ ([NeRF Eq. 3](https://arxiv.org/html/2003.08934)).
2. **EWA projection.** Σ′ = J W Σ Wᵀ Jᵀ ([3DGS Eq. 5](https://arxiv.org/html/2308.04079)). For a camera-space point (x, y, z), the perspective Jacobian is J = [[f/z, 0, −fx/z²], [0, f/z, −fy/z²]] (calculus). So an isotropic splat of std σ on the optical axis projects to a std of fσ/z pixels (math).
3. **Leakage probability under a frame-level random split.** P(location in both splits) = 1 − (1−f)ᵏ − fᵏ, for k frames per location and validation fraction f (math). The fix is a group split by location, then an assertion that no location ID appears in both splits.

#### Exercise (known answer, no GPU)
**Answer (computed here; Monte Carlo agrees):**
- k = 20, f = 0.2: **0.9885** (simulation 0.9889).
- k = 5: 0.6720 (simulation 0.6748).
- k = 100, f = 0.1: ≈ 1.0000.

```python
p_leak=lambda k,f: 1-(1-f)**k-f**k
print(p_leak(20,0.2), p_leak(5,0.2))     # 0.9885, 0.6720
```

**Secondary (two-sample ray):** σ = [1, 2], δ = [0.5, 0.5] → α = [0.3935, 0.6321], T = [1, 0.6065], weights [0.3935, 0.3834], total opacity 0.7769 (computed here; formula from [NeRF](https://arxiv.org/html/2003.08934)).

#### The trap
Believing a "20% random validation split" leaks about 20% of locations. In fact **98.85%** of 20-frame locations appear in both splits, so validation numbers measure memorized places, not generalization (computed here).

### Inferences
- EUVS's three settings map directly onto PhysLoop's validity envelope: translation for lane-change-scale deviations, rotation for heading deviations.
- Instant NuRec's feed-forward speed (about 1.5 s per scene) makes the Stage-5 loop "mine stratum → reconstruct → re-gate" cheap in wall-clock terms.

### Gaps
- No canonical primary source was located for location-leakage-aware splitting of driving datasets, or for CoC auto-labeling. Both are outside the verified set.
- The 3DGS factorization Σ = RSSᵀRᵀ and the default SH degree were not captured verbatim in the text pass and are not quoted here.
- docs/01 C5's EUVS figures (24.6%, 30.6%, PSNR < 15) were not located in this keyword pass. Number verification belongs to other researchers. EUVS's own introduction gives 28%/22%/50% for its Figure-1 example.

## P5 — Simulation and evaluation statistics: sources, derivations, exercise, trap

### Takeaway
The worked examples in docs/01 C6 are correct:
- McNemar, 12 vs 6 discordant pairs: exact two-sided p = 0.2379 (31,180/262,144 one-sided).
- Rule of three: 3/441 = 0.680%, against an exact one-sided 95% Clopper–Pearson bound of 0.677%.
- Spearman (C8): ρ = 0.8.

The canonical statistics sources all resolve. The one correction is that the standard IRT text has a 2025 second edition with a changed title.

### Cited Findings

#### Sources (ranked)
1. **McNemar 1947**, "Note on the sampling error of the difference between correlated proportions or percentages", *Psychometrika* 12(2):153–157. Abstract: "Two formulas are presented for judging the significance of the difference between correlated proportions. The chi square equivalent of one of the developed formulas is pointed out." — [DOI 10.1007/BF02295996](https://doi.org/10.1007/bf02295996)
2. **Clopper & Pearson 1934**, *Biometrika* 26(4):404–413. — [DOI](https://doi.org/10.1093/biomet/26.4.404)
3. **Hanley & Lippman-Hand 1983**, "If nothing goes wrong, is everything all right? Interpreting zero numerators", *JAMA* 249(13):1743–1745. PubMed has no abstract. — [PubMed 6827763](https://pubmed.ncbi.nlm.nih.gov/6827763/); [DOI](https://doi.org/10.1001/jama.1983.03330370053031)
4. **Efron 1979**, "Bootstrap Methods: Another Look at the Jackknife", *Ann. Statist.* 7(1). — [DOI](https://doi.org/10.1214/aos/1176344552)
5. **Cameron, Gelbach & Miller 2008**, "Bootstrap-Based Improvements for Inference with Clustered Errors", *REStat* 90(3):414–427. Working-paper abstract: standard cluster-robust tests "can over-reject, however, with few (5–30) clusters. We investigate inference using cluster bootstrap-t procedures that provide asymptotic refinement." — [DOI 10.1162/rest.90.3.414](https://doi.org/10.1162/rest.90.3.414); [NBER t0344](https://www.nber.org/papers/t0344)
6. **Wald 1945**, "Sequential Tests of Statistical Hypotheses", *Ann. Math. Statist.* 16(2):117–186. — [DOI](https://doi.org/10.1214/aoms/1177731118)
7. **Rasch**, *Probabilistic Models for Some Intelligence and Attainment Tests*, Danish Institute for Educational Research, Copenhagen 1960. Expanded edition University of Chicago Press 1980, with a preface and afterword by B. D. Wright. — [Psychometrika review (Cambridge Core)](https://resolve.cambridge.org/core/journals/psychometrika/article/g-rasch-studies-in-mathematical-psychology-i-probabilistic-models-for-some-intelligence-and-attainment-tests-copenhagen-demnark-the-danish-institute-for-educational-research-1960-pp-184-xiii/1586D1BD7324DC060BA84E6BA439F05B); [GESIS record](https://data.gesis.org/gesiskg/resource/zis-Rasch1960Probabilistic)
8. **Embretson & Reise**, *Item Response Theory: Foundations for Psychologists and Social Scientists*, **2nd ed.**, Routledge, 2025-10-17. The first edition was titled *Item Response Theory (for Psychologists)*; Crossref's Psychology Press e-book record is dated 2013. — [DOI 10.4324/9781315726557](https://doi.org/10.4324/9781315726557); [1st ed. DOI](https://doi.org/10.4324/9781410605269)
9. **Messick 1995**, "Validity of psychological assessment: Validation of inferences from persons' responses and performances as scientific inquiry into score meaning", *Am. Psychol.* 50(9):741–749. — [DOI](https://doi.org/10.1037/0003-066x.50.9.741)
10. **Cronbach & Meehl 1955**, "Construct validity in psychological tests", *Psychol. Bull.* 52(4):281–302. — [DOI](https://doi.org/10.1037/h0040957)
11. **Kalra & Paddock 2016**, RAND RR-1478-RC. Detailed under P7. — [RAND](https://www.rand.org/pubs/research_reports/RR1478.html)
12. **IRT item information, 1PL/Rasch case.** "calculated by multiplying the probability of endorsing a correct response multiplied by the probability of answering incorrectly". — [Columbia Mailman IRT page](https://www.publichealth.columbia.edu/research/population-health-methods/item-response-theory)

#### Derivations to own
1. **Exact McNemar.** Under H₀, the discordant count b ~ Binomial(b + c, ½), and p = 2·P(X ≤ min(b, c)) (math; framing per [McNemar 1947](https://doi.org/10.1007/bf02295996)).
2. **Zero-failure bound.** Solve (1 − p)ⁿ = α to get p_U = 1 − α^{1/n} ≈ −ln α / n = 2.996/n ≈ 3/n. This is a **one-sided** 95% bound, equal to the Clopper–Pearson upper limit at x = 0 (math).
3. **Clustered rollouts.** For n scenes × m seeds with intra-class correlation ρ, Var(mean) = σ²/(nm) · [1 + (m − 1)ρ]. Simulation agrees: 2.236 against formula 2.20 for m = 3, ρ = 0.6 (computed here). Hence bootstrap over scenes or logs, and beware few clusters per stratum ([CGM 2008](https://www.nber.org/papers/t0344)).
4. **IRT 2PL and SPRT.**
   - P(θ) = 1/(1 + e^{−a(θ−b)}); Fisher information a²P(1−P), maximal a²/4 at θ = b (math; the a = 1 case matches the [Columbia](https://www.publichealth.columbia.edu/research/population-health-methods/item-response-theory) description).
   - SPRT: a random walk of the log-likelihood ratio between two boundaries ([Wald 1945](https://doi.org/10.1214/aoms/1177731118); boundary approximations under Gaps).

#### Exercise (known answer, no GPU)
Use docs/01 C6: A fails 30, B fails 24 on 441 paired scenarios, with discordant pairs b = 12 and c = 6.

**Answer (computed here):**
- Σ_{k≤6} C(18,k) = **31,180**, out of 2¹⁸ = 262,144.
- One-sided p = **0.11894**; two-sided **p = 0.23788**.
- Zero failures in 441: 3/441 = 0.680%; exact one-sided 95% = 0.677%; two-sided 95% Clopper–Pearson upper = 0.833%.

```python
from math import comb
lo=sum(comb(18,k) for k in range(7)); print(lo, lo/2**18, 2*lo/2**18)   # 31180 0.11894 0.23788
print(3/441, 1-0.05**(1/441), 1-0.025**(1/441))                          # 0.00680 0.00677 0.00833
```

**How to check:** use scipy.stats.binomtest(6, 18, 0.5) if scipy is available, or exact enumeration as above.

**Secondary:**
- Spearman example from docs/01 C8: Σd² = 4, ρ = 0.8 ✓.
- 2PL with a = 1.7: I_max = 0.7225.
- SPRT with α = 0.05, β = 0.20: A = 16 (log 2.7726), B = 0.2105 (log −1.5581). These use Wald's approximations, which were not text-verified (see Gaps).

#### The trap
- **The uncorrected McNemar χ²** = 2.0 gives **p = 0.157**, which looks much closer to significance.
- **An unpaired two-proportion z-test** (z = 0.843) gives **p = 0.399**, throwing away the pairing.
- The continuity-corrected χ² (p = 0.2386) happens to land close to the exact test.
- **Calling 3/n a "95% CI".** It is one-sided. The two-sided 95% upper limit is 0.833%, not 0.68%.

### Inferences
- The gate's statistics stack can be built entirely from 1934–1995 primary sources, all verified here. No 2026 source is needed for Stage 1a.
- With 441 scenes split into strata, some strata will have few clusters. CGM's "few (5–30) clusters" over-rejection warning applies directly to per-stratum cluster bootstraps.

### Gaps
- **Wald's approximate boundaries** A ≈ (1−β)/α and B ≈ β/(1−α) were *not* verified from text. Wald 1945 is verified only as a record; the Oxford and IISc lecture notes returned 403/404.
- No canonical primary source was located for "variance decomposition (policy / simulator / sampling)" or "data contracts" as framed in docs/03 P5. These read as PhysLoop's own constructs.
- The Messick and Cronbach & Meehl abstracts were not available via Crossref, so their content (the six aspects of validity; the nomological network) is not quoted here.

## P6 — Deployment and real-time: sources, derivations, exercise, trap

### Takeaway
The format arithmetic is fully checkable against primary sources:
- E4M3's maximum is 448: bit pattern S.1111.110 = 1.75 × 2⁸. Under IEEE conventions it would be 240.
- MX formats use 32-element blocks with an E8M0 scale. NVFP4 uses 16-element blocks with an E4M3 scale plus an FP32 per-tensor scale.
- NVFP4's 4.5 effective bits reproduce NVIDIA's own "≈3.5× vs FP16, 1.8× vs FP8" claim.

PMPP's latest edition is the **5th (2026)**, not the 4th.

### Cited Findings

#### Sources (ranked)
1. **Williams, Waterman & Patterson 2009**, "Roofline: An Insightful Visual Performance Model for Multicore Architectures", *CACM* 52(4):65–76. — [DOI 10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785)
2. **Micikevicius et al. 2022 (FP8), Table 1.**
   - E4M3: bias 7, no infinities, NaN = S.1111.111, "Max normal S.1111.110 = 1.75·2⁸ = 448".
   - E5M2: max S.11110.11 = 1.75·2¹⁵ = 57,344, with IEEE special values.
   - "The maximum representable magnitude without this modification would be 240."

   — [arXiv 2209.05433 (HTML)](https://arxiv.org/html/2209.05433)
3. **Rouhani et al. 2023 (MX), Table 1.** MXFP8, MXFP6, MXFP4 (FP4 E2M1) and MXINT8 all use block size 32 with an E8M0 8-bit shared scale. — [arXiv 2310.10537 (HTML)](https://arxiv.org/html/2310.10537)
4. **NVIDIA, "Introducing NVFP4…" (24 Jun 2025).**
   - "a fine-grained E4M3 scaling factor to each 16-value micro-block… a second-level FP32 scalar applied per tensor".
   - MXFP4 is "1 shared power-of-two scale per 32 value block".
   - NVFP4 "cuts model memory footprint by approximately 3.5x relative to FP16 and 1.8x compared to FP8".

   — [NVIDIA Technical Blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
5. **SmoothQuant (Xiao et al.).** s_j = max(|X_j|)^α / max(|W_j|)^{1−α}. "α = 0.5 is a general sweet spot for all the OPT and BLOOM models, and α = 0.75 for GLM-130B". — [arXiv 2211.10438 (HTML)](https://arxiv.org/html/2211.10438)
6. **AWQ** (Lin et al.) — [arXiv 2306.00978](https://arxiv.org/abs/2306.00978); **GPTQ** (Frantar et al.) — [arXiv 2210.17323](https://arxiv.org/abs/2210.17323). Identity only.
7. **CUDA Programming Guide**, §4.2 "CUDA Graphs" (the guide now lives at /cuda-programming-guide/). — [NVIDIA docs](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html). **TensorRT** latest is **11.3.0**, built against CUDA 13.4. — [TensorRT docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html)
8. **Hwu, Kirk & El Hajj**, *Programming Massively Parallel Processors: A Hands-on Approach*, **5th ed.**, Elsevier. The page header says "5th Edition - February 27, 2026, Latest edition"; product details say "Published: June 3, 2026"; ISBN 978-0-443-43900-1. The 4th edition is dated May 28, 2022 (ISBN 978-0-323-91231-0) and its page says "Newer edition is available". — [Elsevier 5e](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-443-43900-1); [Elsevier 4e](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0)
9. **VLA-Perf** (Jiang, Clemons, Sankaralingam, Kozyrakis 2026). An analytical performance model with "15 key takeaways". — [arXiv 2602.18397](https://arxiv.org/abs/2602.18397)
10. **FlashDrive** (2026). On Alpamayo 1.5-10B with W4A8, latency goes "from 717ms to 151ms (4.7x)". It uses streaming KV reuse, diffusion-drafter speculative decoding, adaptive step caching, and "CUDA Graph compilation and kernel fusion". — [arXiv 2608.12932](https://arxiv.org/abs/2608.12932)

#### Derivations to own
1. **Roofline.**
   - Attainable FLOP/s = min(F_peak, AI × BW); the ridge is at AI* = F/BW.
   - For a weight-dominated GEMM with batch B and b bytes per weight, AI = 2B/b, so the crossover batch is **B\* = b·F/(2·BW)**. This matches docs/03 §6 (math).
2. **E4M3 maximum** from the bit pattern (1 + 6/8)·2^{15−7} = 448, and why reserving exponent 1111 would give 1.875·2⁷ = 240 ([FP8 paper](https://arxiv.org/html/2209.05433)).
3. **Block-scaled 4-bit effective bits.**
   - MXFP4 = 4 + 8/32 = **4.25** bits per element; NVFP4 = 4 + 8/16 = **4.5** bits per element (the per-tensor FP32 is negligible).
   - Ratios: FP16/NVFP4 = 3.556×, FP8/NVFP4 = 1.778×, FP16/MXFP4 = 3.765× (math from [MX Table 1](https://arxiv.org/html/2310.10537) and [NVIDIA](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)).
4. **SmoothQuant equivalence.** Y = (X·diag(s)⁻¹)(diag(s)·W). With max|X_j| = 64, max|W_j| = 0.25 and α = 0.5: s = 16, and both maxima become 4 (computed here).

#### Exercise (known answer, no GPU)
Derive NVFP4's memory savings from its definition and check them against NVIDIA's published claim.

**Answer (computed here):** 3.556× vs FP16 and 1.778× vs FP8. NVIDIA reports "approximately 3.5x" and "1.8x". Also 448 / 240 / 57,344 for E4M3 / IEEE-style 4-bit exponent / E5M2.

```python
print(1.75*2**8, 1.875*2**7, 1.75*2**15)    # 448, 240, 57344
mx, nv = 4+8/32, 4+8/16                      # bits/element incl. block scale
print(16/nv, 8/nv, 16/mx)                     # 3.556 1.778 3.765
```

**Roofline example** (hypothetical device: F = 100 TFLOP/s, BW = 1 TB/s): B* = 100 for BF16, 50 for FP8, 28.1 for NVFP4 if F were unchanged (computed here).

**How to check:** the outputs must match the FP8 paper's Table 1 and NVIDIA's ratios.

#### The trap
- **"4-bit is 4× smaller than FP16."** Once block scales are counted, NVFP4 is 3.56× and MXFP4 3.76×.
- **"E4M3 tops out at 240."** That is IEEE-style; the actual maximum is 448.
- **Assuming lower weight precision always moves a batch-1 kernel toward compute-bound.** At fixed F, lowering b lowers B\*. On real silicon F also changes with precision, so recompute the roofline per format.

### Inferences
- docs/03 P6's trap ("VLM quantized but not the memory-bound expert") is consistent with FlashDrive's abstract, which applies W4A8 to the whole model. Whether FlashDrive's expert stays BF16 (docs/01 C10) was not checked; it belongs to the numbers researchers.
- The E2M1 magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6} follow from 1 sign / 2 exponent / 1 mantissa bits ([NVIDIA](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)), *assuming* bias 1 and no Inf/NaN encodings. That assumption was not text-verified.

### Gaps
- The OCP Microscaling Formats specification v1.0 itself was not accessed (it is a PDF). Concrete MX parameters were taken from the MX paper's Table 1, which defers to that spec.
- The CUDA Graphs and TensorRT chapter contents were not read beyond the section listing and the version.

## P7 — Safety and production: sources, derivations, exercise, trap

### Takeaway
RAND's headline figures were confirmed against its own PDF:
- **275 million** failure-free miles (about 12.5 years for a 100-car fleet) to show, with 95% confidence, a fatality rate at most 1.09 per 100M miles.
- **8.8 billion** miles to estimate that rate to within 20%.
- **11 billion** miles to show it is 20% better than humans with 80% power.

That arithmetic is why no simulation gate should ever claim per-mile rates. The standards shelf is ISO 26262:2018, ISO 21448:2022, ISO/PAS 8800:2024, UL 4600 Ed. 3 (2023) and GSN v3 (2021).

### Cited Findings

#### Sources (ranked)
1. **Kalra & Paddock 2016**, *Driving to Safety*, RAND RR-1478-RC (published 12 Apr 2016, 15 pp., DOI 10.7249/RR1478; journal version *Transportation Research Part A* 94:182–193).
   - Eq. 2: n = ln(1 − C)/ln(R).
   - "To demonstrate that fully autonomous vehicles have a fatality rate of 1.09 fatalities per 100 million miles (R = 99.9999989%) with a C = 95% confidence level, the vehicles would have to be driven 275 million failure-free miles… this would take about 12.5 years."
   - Table 1: 8.8 billion miles (400 years) "to within 20% of the true rate", and 11 billion miles (500 years) for 95% confidence and 80% power that the rate is 20% better.
   - Text: ≈5 billion miles at 95% confidence (50% power); a 5% improvement would need 215 billion miles.

   — [RAND PDF](https://www.rand.org/content/dam/rand/pubs/research_reports/RR1400/RR1478/RAND_RR1478.pdf); [landing page](https://www.rand.org/pubs/research_reports/RR1478.html); [TR-A DOI](https://doi.org/10.1016/j.tra.2016.09.010)
2. **Shalev-Shwartz, Shammah & Shashua 2017 (RSS), Lemma 2.**
   - d_min = [v_r ρ + ½ a_max,accel ρ² + (v_r + ρ a_max,accel)²/(2 a_min,brake) − v_f²/(2 a_max,brake)]₊.
   - The parameters "control the soundness/usefulness tradeoff" (Remark 2).

   — [arXiv 1708.06374 (HTML)](https://arxiv.org/html/1708.06374)
3. **ISO 26262:2018** — the 2nd edition, in twelve parts, extending the 2011 first edition to all series-production road vehicles. Secondary sources only; iso.org returned 403 to automated fetches. — [itemis glossary](https://www.itemis.com/en/glossary/iso-26262/); [Omnex: "The New ISO 26262 2nd Edition"](https://my.omnex.com/webinars/the-new-iso-26262-2nd-edition-changes-and-highlights)
4. **ISO 21448:2022 (SOTIF).** Absence of unreasonable risk "due to a hazard caused by functional insufficiencies"; applies to driving-automation levels 1–5; "does not apply to faults covered by the ISO 26262 series". — [iso.org 77490](https://www.iso.org/standard/77490.html); [ANSI preview](https://webstore.ansi.org/preview-pages/ISO/preview_ISO+21448-2022.pdf)
5. **ISO/PAS 8800:2024**, "Road vehicles — Safety and artificial intelligence". It covers "the risk of undesired safety-related behaviour at the vehicle level due to output insufficiencies, systematic errors and random hardware errors of AI elements", focuses on ML, and runs to 172 pp. A mirror lists approval on 13 Dec 2024. — [iso.org/fr 83303](https://www.iso.org/fr/standard/83303.html); [SCC record](https://scc-ccn.ca/standardsdb/standards/8187375)
6. **UL 4600 Edition 3** (released 17 Mar 2023). A goal-based, technology-agnostic safety-case standard. Ed. 3 adds autonomous trucking, a more precise definition of safety performance indicator (SPI), and post-incident behaviour requirements. — [UL news](https://www.ul.com/news/ul-4600-edition-3-updates-incorporate-autonomous-trucking); [ULSE](https://ulse.org/ul-standards-engagement/autonomous-vehicle-technology/autonomous-trucking-incorporated-third)
7. **Koopman**, *How Safe Is Safe Enough? Measuring and Predicting Autonomous Vehicle Safety* (published 2 Sep 2022). It covers risk acceptance, safety cases, SPIs and deployment decisions. — [Goodreads record](https://www.goodreads.com/book/show/64663237); [UW library catalog](https://search.library.wisc.edu/catalog/9913654221802121)
8. **GSN Community Standard Version 3** (SCSC-141C, May 2021), maintained by the SCSC Assurance Case Working Group. — [scsc.uk citation](https://scsc.uk/resources/citation_r1386.html); [scsc.uk/scsc-141c](https://scsc.uk/scsc-141c)
9. **Favaro et al. 2025 (Waymo).** Readiness determination as risk assessment; "informed by Waymo's experience". — [arXiv 2505.09880](https://arxiv.org/abs/2505.09880)
10. **NVIDIA Halos.** "a full-stack, comprehensive safety system that unifies safety elements across vehicle architecture, AI models, chips, software, tools, and services". — [NVIDIA Halos AV page](https://www.nvidia.com/en-us/ai-trust-center/halos/autonomous-vehicles/)

#### Derivations to own
1. **RAND Eq. 2.** Zero failures in n miles at per-mile reliability R has probability Rⁿ = 1 − C, so n = ln(1−C)/ln R ≈ 3/λ at C = 95%, the same rule of three as in P5 ([RAND](https://www.rand.org/content/dam/rand/pubs/research_reports/RR1400/RR1478/RAND_RR1478.pdf)).
2. **RAND power formula** (Eq. 9): n = λ_alt·((z_{1−α} + z_{1−β})/(λ₀ − λ_alt))². It gives 11.34 billion miles for A = 20%.
3. **RSS d_min.** Worst-case kinematics: the rear car accelerates at a_max,accel for ρ seconds, then brakes at a_min,brake; the front car brakes at a_max,brake ([Lemma 2](https://arxiv.org/html/1708.06374)).

#### Exercise (known answer, no GPU)
Reproduce RAND's 275 million miles and 12.5 years.

**Answer (computed here):** **274.8 million miles** and **12.55 years**, against RAND's printed "275 million" and "about 12.5 years".

```python
import math
n=math.log(1-0.95)/math.log(1-1.09e-8); print(n/1e6, n/(100*24*365*25))   # 274.8, 12.55
def rss(vr,vf,rho,acc,bmin,bmax): return max(0,vr*rho+.5*acc*rho**2+(vr+rho*acc)**2/(2*bmin)-vf**2/(2*bmax))
print(rss(20,20,.5,3,4,8), rss(20,20,.5,3,4,4))   # 43.16 m, 18.16 m (trap)
```

**Secondary (RSS):** v_r = v_f = 20 m/s, ρ = 0.5 s, a_max,accel = 3, a_min,brake = 4, a_max,brake = 8 m/s² gives **d_min = 43.16 m** (computed here; the parameters are illustrative).

#### The trap
- **"Zero fatalities in 10 million test miles shows the AV is safer than humans."** The 95% upper bound is 3/10⁷ = **30 per 100M miles**, about 27.5× the human 1.09 (computed here).
- **The RSS shortcut:** assuming the lead car brakes no harder than you (a_max,brake = a_min,brake = 4) gives **18.16 m**, less than half the RSS value. A 2-second rule gives 40 m.

### Inferences
- RAND's arithmetic is the quantitative backbone of docs/02 §5's "never claim safe or per-mile rates". A 441-scene gate is many orders of magnitude short of fatality-rate evidence, so its claims must stay comparative and paired.
- Three things map naturally onto the G4 guardrail and the release note:
  - UL 4600's safety-case approach and SPI definition;
  - ISO/PAS 8800's framing of AI "output insufficiencies";
  - SOTIF's "functional insufficiencies".

### Gaps
- UN R171 (DCAS) and UNECE NATM (WP.29-187-08) are cited in docs/03 but were outside this assignment's list and were not checked.
- None of the ISO texts were readable (paywalled, and iso.org returned 403 to bots). Clause-level requirements for ML validation are not stated here.

## Reference-shelf identity check and full verification ledger

### Takeaway
**Every arXiv ID on docs/03 §7's shelf and in docs/01 §3 resolves to the claimed title, authors and year: 13/13 IDs plus EUVS.** No IDs were wrong and nothing was NOT FOUND. Corrections are limited to:
- a shelf categorization (RoaD is closed-loop *supervised* fine-tuning, not RL);
- EUVS lacking an ID (it is arXiv 2412.05256);
- two editions in the brief's sources: PMPP is now in its 5th edition (2026, with Hwu as first author), and Embretson & Reise's 2nd edition (2025) has a new title.

### Cited Findings

#### A. docs reference shelf (docs/03 §7; docs/01 §2–§3)

| # | Reference as written (doc + §) | Status | Exact title / authors / year found | Correct ID if different | URL | Accessed |
|---|---|---|---|---|---|---|
| A1 | "Alpamayo-R1 (arXiv 2511.00088)" — 03 §7; 01 C1 | VERIFIED | *Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail* — NVIDIA (corporate) + Yan Wang, Wenjie Luo et al. (44 author entries); v1 2025-10-30, v2 2026-01-07 | — | https://arxiv.org/abs/2511.00088 | 2026-10-03 |
| A2 | "HF model card nvidia/Alpamayo-1.5-10B" — 01 C1 | VERIFIED | Alpamayo 1.5 card: base_model nvidia/Cosmos-Reason2-8B; backbone 8.2B + action expert 2.3B; HF release 03/19/2026 | — | https://huggingface.co/nvidia/Alpamayo-1.5-10B | 2026-10-03 |
| A3 | "Flow Matching Guide and Code (arXiv 2412.06264)", Lipman et al. — 03 §7, §4 | VERIFIED | *Flow Matching Guide and Code* — Lipman, Havasi, Holderrieth, Shaul + 6 (10 authors); 2024-12-09 | — | https://arxiv.org/abs/2412.06264 | 2026-10-03 |
| A4 | "GRPO (DeepSeekMath, arXiv 2402.03300)" — 03 §7, §4 | VERIFIED | *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* — Shao, Wang, Zhu, Xu + 7; 2024-02-05 (v3 2024-04-27) | — | https://arxiv.org/abs/2402.03300 | 2026-10-03 |
| A5 | "OPTED (arXiv 2609.20756)"; "privileged-teacher RL (OPTED)" — 03 §7, P3 | VERIFIED | *OPTED: On-Policy Fine-Tuning for End-to-End Driving using a Render-Free Teacher* — Da Col, Igl, Karkus, Chitta + 4; 2026-09-17. An RL-trained privileged teacher on vectorized inputs supervises the student, which matches "privileged-teacher RL" | — | https://arxiv.org/abs/2609.20756 | 2026-10-03 |
| A6 | "RoaD (arXiv 2512.01993)" listed under "RL" — 03 §7 | VERIFIED | *RoaD: Rollouts as Demonstrations for Closed-Loop Supervised Fine-Tuning of Autonomous Driving Policies* — Garcia-Cobo, Igl, Karkus, Zhang + 4; 2025-12-01. This is closed-loop **supervised** fine-tuning, not RL | — | https://arxiv.org/abs/2512.01993 | 2026-10-03 |
| A7 | "nuPlan misalignment (arXiv 2306.07962)" — 03 §7; 01 C3 | VERIFIED | *Parting with Misconceptions about Learning-based Vehicle Motion Planning* — Dauner, Hallgarten, Geiger, Chitta; 2023-06-13 (v2 2023-11-02) | — | https://arxiv.org/abs/2306.07962 | 2026-10-03 |
| A8 | "pseudo-simulation (arXiv 2506.04218)" — 03 §7; 01 C8 | VERIFIED | *Pseudo-Simulation for Autonomous Driving* — Cao, Hallgarten, Li, Dauner + 10; 2025-06-04 (v3 2026-03-20) | — | https://arxiv.org/abs/2506.04218 | 2026-10-03 |
| A9 | "Bosch cross-benchmark (arXiv 2605.00066)"; "Bosch 2026" — 03 §7, §1 P1; 01 C3, C8 | VERIFIED | *Do Open-Loop Metrics Predict Closed-Loop Driving? A Cross-Benchmark Correlation Study of NAVSIM and Bench2Drive* — Yiru Wang, Anqing Jiang, Shuo Wang, Yuwen Heng, Hai Yang, Yang Chen, Hao Sun (all Bosch Corporate Research, Shanghai); 2026-04-30 | — | https://arxiv.org/abs/2605.00066 | 2026-10-03 |
| A10 | "Waymo approval guidelines (arXiv 2505.09880)"; "Favaro et al." — 03 §7; 01 §2 | VERIFIED | *Determining Absence of Unreasonable Risk: Approval Guidelines for an Automated Driving System Deployment* — Favaro, Schnelle, Fraade-Blanar, Victor + 5; 2025-05-15 (v2 2025-05-29); "informed by Waymo's experience" | — | https://arxiv.org/abs/2505.09880 | 2026-10-03 |
| A11 | "RAND Driving to Safety (RR1478)" — 03 §7 | VERIFIED | Kalra & Paddock, *Driving to Safety: How Many Miles of Driving Would It Take to Demonstrate Autonomous Vehicle Reliability?*, RR-1478-RC, RAND 2016 (12 Apr 2016), 15 pp.; also *Transp. Res. A* 94:182–193 (2016) | DOI 10.7249/RR1478 | https://www.rand.org/pubs/research_reports/RR1478.html | 2026-10-03 |
| A12 | "Instant NuRec (arXiv 2607.14203)" — 03 §7 | VERIFIED | *Instant NuRec: Feed-Forward 3D Gaussian Reconstruction for Driving Scene Simulation* — NVIDIA + Jiahui Huang, Jiawei Ren + 13; 2026-07-15 | — | https://arxiv.org/abs/2607.14203 | 2026-10-03 |
| A13 | "OmniDreams (arXiv 2606.03159)" — 03 §7 | VERIFIED | *NVIDIA OmniDreams: Real-Time Generative World Model for Closed-Loop Autonomous Vehicle Simulation* — Basant, Kar, Paschalidou, Wei + 29; 2026-06-02 (v3 2026-09-23) | — | https://arxiv.org/abs/2606.03159 | 2026-10-03 |
| A14 | "EUVS novel-view limits (ICCV 2025)" — 03 §7; 01 C5 | VERIFIED | *Extrapolated Urban View Synthesis Benchmark* — Xiangyu Han, Zhen Jia, Boyi Li, Yan Wang + 7 (NYU, NVIDIA, UPenn, USC, Stanford). The project page and BibTeX give ICCV 2025 | arXiv 2412.05256 (the doc gives no ID) | https://ai4ce.github.io/EUVS-Benchmark/ | 2026-10-03 |
| A15 | "VLA-Perf (arXiv 2602.18397)" — 03 §7; 01 C10 | VERIFIED | *How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf* — Wenqi Jiang, Jason Clemons, Karu Sankaralingam, Christos Kozyrakis; 2026-02-20 | — | https://arxiv.org/abs/2602.18397 | 2026-10-03 |
| A16 | "FlashDrive (arXiv 2608.12932)" — 03 §7; 01 C10 | VERIFIED | *FlashDrive: Flash Vision-Language-Action Inference for Autonomous Driving* — Zekai Li, Yihao Liang, Hongfei Zhang, Jian Chen + 2; 2026-08-13 | — | https://arxiv.org/abs/2608.12932 | 2026-10-03 |
| A17 | "LaValle, Planning Algorithms" — 03 §7 (classical) | VERIFIED | Steven M. LaValle, *Planning Algorithms*, Cambridge Univ. Press 2006, 842 pp., ISBN 9780521862059; full HTML free | DOI 10.1017/CBO9780511546877 | https://lavalle.pl/planning/ | 2026-10-03 |
| A18 | "Thrun et al., Probabilistic Robotics" — 03 §7 (classical) | VERIFIED | Thrun, Burgard, Fox, *Probabilistic Robotics*, MIT Press 2005, 647 pp. (per review record; the MIT Press page returned 403) | — | https://doi.org/10.1162/artl.2008.14.2.227 | 2026-10-03 |
| A19 | "Sutton & Barto" — 03 §7 (classical) | VERIFIED | Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press 2018; full PDF on the author's site | — | http://incompleteideas.net/book/the-book-2nd.html | 2026-10-03 |
| A20 | "Ross et al., DAgger (2011)" — 03 §7 (classical) | VERIFIED | Ross, Gordon, Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*, AISTATS 2011, PMLR 15:627–635 | arXiv 1011.0686 | https://proceedings.mlr.press/v15/ross11a.html | 2026-10-03 |
| A21 | "Halos" — 03 §7 Safety | VERIFIED | NVIDIA Halos: "full-stack, comprehensive safety system…" (AV page) | — | https://www.nvidia.com/en-us/ai-trust-center/halos/autonomous-vehicles/ | 2026-10-03 |

#### B. Pillar primary sources named in the task brief

| # | Reference as written (location) | Status | Exact title / authors / year found | Correct ID if different | URL | Accessed |
|---|---|---|---|---|---|---|
| B1 | Kong et al. 2015 IEEE IV — brief P1 | VERIFIED | Kong, Pfeiffer, Schildbach, Borrelli, "Kinematic and dynamic vehicle models for autonomous driving control design", IEEE IV 2015, pp. 1094–1099 | DOI 10.1109/IVS.2015.7225830 | https://doi.org/10.1109/IVS.2015.7225830 | 2026-10-03 |
| B2 | Rajamani, Vehicle Dynamics and Control — brief P1 | VERIFIED | Rajesh Rajamani, 2nd ed., Springer (Mech. Eng. Series), © 2012 (hardcover 21 Dec 2011), "Latest edition" | DOI 10.1007/978-1-4614-1433-9 | https://link.springer.com/book/10.1007/978-1-4614-1433-9 | 2026-10-03 |
| B3 | Werling et al. 2010 — brief P1 | VERIFIED | Werling, Ziegler, Kammel, Thrun, "Optimal trajectory generation for dynamic street scenarios in a Frenét Frame", ICRA 2010, pp. 987–993 | DOI 10.1109/ROBOT.2010.5509799 | https://doi.org/10.1109/ROBOT.2010.5509799 | 2026-10-03 |
| B4 | EPDMS — brief P1 | VERIFIED | Defined in *Pseudo-Simulation for AD* (builds on PDMS; cites NAVSIM v2 as [44]) | arXiv 2506.04218 | https://arxiv.org/html/2506.04218 | 2026-10-03 |
| B5 | NAVSIM (context for PDMS) — brief P1 | VERIFIED | *NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking* — Dauner et al.; 2024-06-21 | arXiv 2406.15349 | https://arxiv.org/abs/2406.15349 | 2026-10-03 |
| B6 | Vaswani 2017 — brief P2 | VERIFIED | *Attention Is All You Need* — Vaswani + 7; 2017-06-12 | arXiv 1706.03762 | https://arxiv.org/abs/1706.03762 | 2026-10-03 |
| B7 | Dosovitskiy 2020 — brief P2 | VERIFIED | *An Image is Worth 16x16 Words…* — Dosovitskiy + 11; 2020-10-22 | arXiv 2010.11929 | https://arxiv.org/abs/2010.11929 | 2026-10-03 |
| B8 | LLaVA — brief P2 | VERIFIED | *Visual Instruction Tuning* — Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee; 2023-04-17 | arXiv 2304.08485 | https://arxiv.org/abs/2304.08485 | 2026-10-03 |
| B9 | Qwen2.5-VL technical report — brief P2 | VERIFIED | *Qwen2.5-VL Technical Report* — Bai et al. (27 authors); 2025-02-19 | arXiv 2502.13923 | https://arxiv.org/abs/2502.13923 | 2026-10-03 |
| B10 | (added) Qwen3-VL report; Cosmos-Reason2 base — P2 | VERIFIED | *Qwen3-VL Technical Report* — Bai et al. (64); 2025-11-26. "Cosmos-Reason2 is based on the Qwen3-VL architecture" (README) | arXiv 2511.21631 | https://arxiv.org/abs/2511.21631 ; https://github.com/nvidia-cosmos/cosmos-reason2 | 2026-10-03 |
| B11 | FAST, arXiv 2501.09747 — brief P2 | VERIFIED | *FAST: Efficient Action Tokenization for Vision-Language-Action Models* — Pertsch, Stachowicz, Ichter, Driess + 5; 2025-01-16 | — | https://arxiv.org/abs/2501.09747 | 2026-10-03 |
| B12 | Lipman et al. 2022, arXiv 2210.02747 — brief P2 | VERIFIED | *Flow Matching for Generative Modeling* — Lipman, Chen, Ben-Hamu, Nickel, Le; 2022-10-06 | — | https://arxiv.org/abs/2210.02747 | 2026-10-03 |
| B13 | Rectified flow, arXiv 2209.03003 — brief P2 | VERIFIED | *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow* — Xingchao Liu, Chengyue Gong, Qiang Liu; 2022-09-07 | — | https://arxiv.org/abs/2209.03003 | 2026-10-03 |
| B14 | pi0, arXiv 2410.24164 — brief P2 | VERIFIED | *π0: A Vision-Language-Action Flow Model for General Robot Control* — Black, Brown, Driess, Esmail + 20; 2024-10-31 (v4 2026-01-08) | — | https://arxiv.org/abs/2410.24164 | 2026-10-03 |
| B15 | Ross & Bagnell 2010 (BC bound) — P3 | VERIFIED | *Efficient Reductions for Imitation Learning* — Ross, Bagnell; AISTATS 2010, PMLR 9:661–668 | — | https://proceedings.mlr.press/v9/ross10a.html | 2026-10-03 |
| B16 | PPO, arXiv 1707.06347 — brief P3 | VERIFIED | *Proximal Policy Optimization Algorithms* — Schulman, Wolski, Dhariwal, Radford, Klimov; 2017-07-20 | — | https://arxiv.org/abs/1707.06347 | 2026-10-03 |
| B17 | Schulman "Approximating KL Divergence" — brief P3 | VERIFIED | John Schulman blog post (cited as "Schulman, 2020" by DeepSeekMath); k1/k2/k3 table | — | http://joschu.net/blog/kl-approx.html | 2026-10-03 |
| B18 | Hinton et al., arXiv 1503.02531 — brief P3 | VERIFIED | *Distilling the Knowledge in a Neural Network* — Hinton, Vinyals, Dean; 2015-03-09 | — | https://arxiv.org/abs/1503.02531 | 2026-10-03 |
| B19 | GKD, arXiv 2306.13649 — brief P3 | VERIFIED | *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* — Agarwal, Vieillard, Zhou, Stanczyk + 3; 2023-06-23 (v3 2024-01-17). GKD is the method name | — | https://arxiv.org/abs/2306.13649 | 2026-10-03 |
| B20 | Flow-GRPO — brief P3 | VERIFIED | *Flow-GRPO: Training Flow Matching Models via Online RL* — Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li + 5; 2025-05-08 | arXiv 2505.05470 | https://arxiv.org/abs/2505.05470 | 2026-10-03 |
| B21 | pi_RL — brief P3 | VERIFIED | *π_RL: Online RL Fine-tuning for Flow-based Vision-Language-Action Models* — Kang Chen, Zhihao Liu, Tonghe Zhang, Zhen Guo + 10; 2025-10-29 | arXiv 2510.25889 | https://arxiv.org/abs/2510.25889 | 2026-10-03 |
| B22 | 3DGS, arXiv 2308.04079 — brief P4 | VERIFIED | *3D Gaussian Splatting for Real-Time Radiance Field Rendering* — Kerbl, Kopanas, Leimkühler, Drettakis; ACM TOG 42(4), 2023 | — | https://arxiv.org/abs/2308.04079 | 2026-10-03 |
| B23 | NeRF — brief P4 | VERIFIED | *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis* — Mildenhall, Srinivasan, Tancik, Barron + 2; 2020-03-19 | arXiv 2003.08934 | https://arxiv.org/abs/2003.08934 | 2026-10-03 |
| B24 | McNemar 1947 — brief P5 | VERIFIED | *Psychometrika* 12(2):153–157 | DOI 10.1007/BF02295996 | https://doi.org/10.1007/bf02295996 | 2026-10-03 |
| B25 | Efron 1979 — brief P5 | VERIFIED | "Bootstrap Methods: Another Look at the Jackknife", *Ann. Statist.* 7(1) | DOI 10.1214/aos/1176344552 | https://doi.org/10.1214/aos/1176344552 | 2026-10-03 |
| B26 | Cameron, Gelbach & Miller 2008 — brief P5 | VERIFIED | *REStat* 90(3):414–427 (Aug 2008); NBER TWP t0344 (2007) | DOI 10.1162/rest.90.3.414 | https://doi.org/10.1162/rest.90.3.414 | 2026-10-03 |
| B27 | Clopper & Pearson 1934 — brief P5 | VERIFIED | *Biometrika* 26(4):404–413 | DOI 10.1093/biomet/26.4.404 | https://doi.org/10.1093/biomet/26.4.404 | 2026-10-03 |
| B28 | Hanley & Lippman-Hand 1983, JAMA — brief P5 | VERIFIED | "If nothing goes wrong, is everything all right? Interpreting zero numerators", *JAMA* 249(13):1743–1745 | PMID 6827763 | https://pubmed.ncbi.nlm.nih.gov/6827763/ | 2026-10-03 |
| B29 | Rasch — brief P5 | VERIFIED | *Probabilistic Models for Some Intelligence and Attainment Tests*, Danish Inst. for Educational Research 1960; expanded ed. Univ. of Chicago Press 1980 | — | https://data.gesis.org/gesiskg/resource/zis-Rasch1960Probabilistic | 2026-10-03 |
| B30 | "a standard text such as Embretson & Reise" — brief P5 | CORRECTED | Edition and title corrected: *Item Response Theory: Foundations for Psychologists and Social Scientists*, **2nd ed.**, Routledge, 2025-10-17 (1st ed. *Item Response Theory (for Psychologists)*; Crossref e-book record dated 2013) | DOI 10.4324/9781315726557 | https://doi.org/10.4324/9781315726557 | 2026-10-03 |
| B31 | Kalra & Paddock 2016, RR-1478 — brief P5 | VERIFIED | See A11; PDF Table 1 confirmed (275M / 8.8B / 11B miles) | — | https://www.rand.org/content/dam/rand/pubs/research_reports/RR1400/RR1478/RAND_RR1478.pdf | 2026-10-03 |
| B32 | Messick 1995 — brief P5 | VERIFIED | *Am. Psychol.* 50(9):741–749 | DOI 10.1037/0003-066X.50.9.741 | https://doi.org/10.1037/0003-066x.50.9.741 | 2026-10-03 |
| B33 | Cronbach & Meehl 1955 — brief P5 | VERIFIED | *Psychol. Bull.* 52(4):281–302 | DOI 10.1037/h0040957 | https://doi.org/10.1037/h0040957 | 2026-10-03 |
| B34 | Wald's SPRT — brief P5 | VERIFIED | "Sequential Tests of Statistical Hypotheses", *Ann. Math. Statist.* 16(2):117–186 (1945); boundary formulas not text-verified | DOI 10.1214/aoms/1177731118 | https://doi.org/10.1214/aoms/1177731118 | 2026-10-03 |
| B35 | Roofline, Williams, Waterman & Patterson 2009 CACM — brief P6 | VERIFIED | *CACM* 52(4):65–76 | DOI 10.1145/1498765.1498785 | https://doi.org/10.1145/1498765.1498785 | 2026-10-03 |
| B36 | FP8, arXiv 2209.05433 — brief P6 | VERIFIED | *FP8 Formats for Deep Learning* — Micikevicius, Stosic, Burgess, Cornea + 11; 2022-09-12 | — | https://arxiv.org/abs/2209.05433 | 2026-10-03 |
| B37 | OCP Microscaling spec — brief P6 | UNVERIFIABLE | Spec not accessed. The MX paper (Rouhani et al. 2023) gives Table 1 and defers to "OCP Microscaling Specification [6]" | MX paper arXiv 2310.10537 (VERIFIED) | https://arxiv.org/abs/2310.10537 | 2026-10-03 |
| B38 | NVIDIA NVFP4 materials — brief P6 | VERIFIED | "Introducing NVFP4 for Efficient and Accurate Low-Precision Inference", NVIDIA Technical Blog, 24 Jun 2025 (Alvarez, Almog, Chung, Layton, Stosic, Krashinsky, Aubrey) | — | https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ | 2026-10-03 |
| B39 | SmoothQuant — brief P6 | VERIFIED | *SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs* — Xiao, Lin, Seznec, Wu + 2; 2022-11-18 | arXiv 2211.10438 | https://arxiv.org/abs/2211.10438 | 2026-10-03 |
| B40 | AWQ — brief P6 | VERIFIED | *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* — Ji Lin et al. (10); 2023-06-01 (v6 2026-04-25) | arXiv 2306.00978 | https://arxiv.org/abs/2306.00978 | 2026-10-03 |
| B41 | GPTQ — brief P6 | VERIFIED | *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* — Frantar, Ashkboos, Hoefler, Alistarh; 2022-10-31 | arXiv 2210.17323 | https://arxiv.org/abs/2210.17323 | 2026-10-03 |
| B42 | CUDA graphs — brief P6 | VERIFIED | CUDA Programming Guide §4.2 "CUDA Graphs" (URL moved to /cuda-programming-guide/) | — | https://docs.nvidia.com/cuda/cuda-programming-guide/index.html | 2026-10-03 |
| B43 | TensorRT — brief P6 | VERIFIED | NVIDIA TensorRT docs; latest release 11.3.0 (built on CUDA 13.4) | — | https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html | 2026-10-03 |
| B44 | "Programming Massively Parallel Processors (Kirk & Hwu, latest edition)" — brief P6 | CORRECTED | Edition and author order corrected: **5th ed.**, Hwu, Kirk & El Hajj, Elsevier, 2026 (header 27 Feb 2026; "Published: June 3, 2026"), ISBN 978-0-443-43900-1. The 4th ed. (22 May 2022) is superseded | ISBN 978-0-443-43900-1 | https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-443-43900-1 | 2026-10-03 |
| B45 | ISO 26262 — brief P7 | VERIFIED | Secondary sources only (iso.org returned 403): ISO 26262:2018, 2nd edition, 12 parts (1st ed. 2011) | — | https://www.itemis.com/en/glossary/iso-26262/ | 2026-10-03 |
| B46 | ISO 21448 (SOTIF) — brief P7 | VERIFIED | ISO 21448:2022, *Road vehicles — Safety of the intended functionality* | ISO record 77490 | https://www.iso.org/standard/77490.html | 2026-10-03 |
| B47 | ISO/PAS 8800 — brief P7 | VERIFIED | ISO/PAS 8800:2024, *Road vehicles — Safety and artificial intelligence*, 172 pp. (approved 13 Dec 2024 per mirror) | ISO record 83303 | https://www.iso.org/fr/standard/83303.html | 2026-10-03 |
| B48 | UL 4600 — brief P7 | VERIFIED | UL 4600, *Standard for Safety for the Evaluation of Autonomous Products*, Edition 3, 17 Mar 2023 | — | https://www.ul.com/news/ul-4600-edition-3-updates-incorporate-autonomous-trucking | 2026-10-03 |
| B49 | Koopman, How Safe Is Safe Enough? (2022) — brief P7 | VERIFIED | Secondary sources (Goodreads, UW library catalog): Philip Koopman, *How Safe Is Safe Enough? Measuring and Predicting Autonomous Vehicle Safety*, 2 Sep 2022 | — | https://www.goodreads.com/book/show/64663237 | 2026-10-03 |
| B50 | Safety cases with GSN — brief P7 | VERIFIED | *Goal Structuring Notation Community Standard (Version 3)*, SCSC-141C, May 2021 | — | https://scsc.uk/resources/citation_r1386.html | 2026-10-03 |
| B51 | RSS, arXiv 1708.06374 — brief P7 | VERIFIED | *On a Formal Model of Safe and Scalable Self-driving Cars* — Shalev-Shwartz, Shammah, Shashua; 2017-08-21 (v6 2018-10-27) | — | https://arxiv.org/abs/1708.06374 | 2026-10-03 |

#### C. Learning-method sources

| # | Reference as written (location) | Status | Exact title / authors / year found | Correct ID if different | URL | Accessed |
|---|---|---|---|---|---|---|
| C1 | Roediger & Karpicke 2006 — brief | VERIFIED | "Test-enhanced learning: taking memory tests improves long-term retention", *Psychol. Sci.* 17(3):249–255 | PMID 16507066 | https://pubmed.ncbi.nlm.nih.gov/16507066/ | 2026-10-03 |
| C2 | Generation effect — brief | VERIFIED | Slamecka & Graf 1978, *JEP: HLM* 4(6):592–604; meta-analysis Bertsch et al. 2007, *Mem. Cognit.* 35(2):201–210 | DOI 10.1037/0278-7393.4.6.592; PMID 17645161 | https://pubmed.ncbi.nlm.nih.gov/17645161/ | 2026-10-03 |
| C3 | Worked-example effect — brief | VERIFIED | Identity only (abstracts not retrieved): Sweller & Cooper 1985, "The Use of Worked Examples as a Substitute for Problem Solving in Learning Algebra", *Cogn. Instr.* 2(1):59–89; Kalyuga et al. 2003, "The Expertise Reversal Effect", *Educ. Psychol.* 38(1):23–31 | — | https://doi.org/10.1207/s1532690xci0201_3 | 2026-10-03 |
| C4 | Teach-back — brief | VERIFIED | Used as teach-back proxies: Chi et al. 1989, "Self-Explanations…", *Cogn. Sci.* 13(2):145–182; Fiorella & Mayer 2013, "The relative benefits of learning by teaching and teaching expectancy", *Contemp. Educ. Psychol.* 38(4):281–288 | — | https://doi.org/10.1207/s15516709cog1302_1 | 2026-10-03 |
| C5 | (added) Dunlosky et al. 2013 | VERIFIED | "Improving Students' Learning With Effective Learning Techniques", *PSPI* 14(1):4–58 | PMID 26173288 | https://pubmed.ncbi.nlm.nih.gov/26173288/ | 2026-10-03 |
| C6 | (added) Karpicke & Blunt 2011 | VERIFIED | Finding contested: *Science* 331(6018):772–775; comment by Mintzes et al., *Science* 334:453 (2011) | — | https://pubmed.ncbi.nlm.nih.gov/22034416/ | 2026-10-03 |
| C7 | "Anthropic study on AI assistance and skill formation (claimed arXiv 2601.20245)" — brief | VERIFIED | *How AI Impacts Skill Formation* — Judy Hanwen Shen (Anthropic Fellows Program), Alex Tamkin (Anthropic); 2026-01-28 (v2 2026-02-01) | — | https://arxiv.org/abs/2601.20245 | 2026-10-03 |
| C8 | (added) Bastani et al. 2025 | VERIFIED | "Generative AI without guardrails can harm learning: Evidence from high school mathematics", *PNAS* 122(26):e2422633122 (erratum 122(34)) | — | https://pubmed.ncbi.nlm.nih.gov/40560616/ | 2026-10-03 |

#### D. Courses (latest year as of 2026-10-03)

| # | Course (location) | Status | Found | URL | Accessed |
|---|---|---|---|---|---|
| D1 | Tübingen Self-Driving Cars (A. Geiger) — brief | VERIFIED | ML-4340, flipped classroom with YouTube lectures; latest run **Winter 2025/26** (exam 13 Feb 2026) | https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/autonomous-vision/lectures/self-driving-cars/ | 2026-10-03 |
| D2 | Berkeley CS285 — brief | VERIFIED | Now "CS 185/285 Deep RL" (S. Levine); current offering **Spring 2026** (staff list cs285-staff-sp2026); public recordings from Fall 2023 | https://rail.eecs.berkeley.edu/deeprlcourse/ | 2026-10-03 |
| D3 | Stanford CS231n — brief | VERIFIED | **Spring 2026** (Fei-Fei Li, Ehsan Adeli, Justin Johnson, Zane Durante, Tiange Xiang); lecture-videos tab | https://cs231n.stanford.edu/ | 2026-10-03 |
| D4 | Stanford CS336 — brief | VERIFIED | Language Modeling from Scratch, **Spring 2026** (staff list cs336-spr2526; Hashimoto, Liang); YouTube playlist | https://cs336.stanford.edu/ | 2026-10-03 |
| D5 | CMU 16-831 — brief | UNVERIFIABLE | Official course page not located in two searches | — | 2026-10-03 |
| D6 | (added) MIT 6.S184 Flow Matching & Diffusion | VERIFIED | **2026** edition (lecture 1 dated 2026-01-20); notes: Holderrieth & Erives, arXiv 2506.02070 | https://diffusion.csail.mit.edu/2026/ | 2026-10-03 |
| D7 | (added) MIT 6.5940 TinyML & Efficient AI (S. Han) | VERIFIED | "6.5940 • Fall • 2026" | https://hanlab.mit.edu/course | 2026-10-03 |
| D8 | (added) CMU 16-745 Optimal Control (Z. Manchester) | VERIFIED | Identity verified; year UNVERIFIABLE (not printed). Live page with convex-MPC/iLQR lectures; no year printed | https://optimalcontrol.ri.cmu.edu/ | 2026-10-03 |

### Inferences
- The earlier AI sessions that wrote docs/03 §7 got every arXiv ID right, including all 2026 IDs. The residual risk is in *numbers and attributions inside* those papers, which other researchers own, not in the IDs.
- docs/03 §7 marks its classical texts "standard, not re-verified". All four are now verified (A17–A20) and can drop that caveat.

### Gaps
- Shelf items outside the assigned parenthetical list were **not checked**:
  - Alpamayo 2 Super blog; NVlabs alpamayo-recipes, alpasim and alpagym repos; the closed-loop post-training blog;
  - facebookresearch/flow_matching; NuRec docs; Jetson Thor specs; the TensorRT-Edge-LLM Alpamayo pipeline;
  - UNECE NATM (WP.29-187-08); the UL summary of ISO/PAS 8800.
- Numbers quoted in docs/01 and docs/03 (for example ρ values, EUVS percentages, latency figures) were not systematically checked here. Incidental confirmations: Bosch ρ = −0.36 (p = 0.43) and ρ = 0.90 (n = 8); Alpamayo 1.5 at 8.2B + 2.3B.

## Current free courses and lecture series by pillar (latest year)

### Takeaway
Every pillar except P5 (measurement statistics) and P7 (safety) has a strong, free, currently offered university course. Those two are best served by primary texts: RAND, Koopman, UL 4600, IRT books. CMU 16-831 could not be verified.

### Cited Findings
- **P1, end-to-end driving systems:** Tübingen *Self-Driving Cars* (Geiger), WS 2025/26. It aims to give "understanding of the entire system comprising perception, learning and vehicle control" and to have students "implement and train simple models for sensori-motor control"; lectures are on YouTube. — [Tübingen](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/autonomous-vision/lectures/self-driving-cars/)
- **P1, MPC and trajectory optimization:** CMU 16-745 (Manchester), covering convex MPC, iLQR/DDP and direct collocation; year not printed. — [16-745](https://optimalcontrol.ri.cmu.edu/). Book: LaValle, free HTML. — [lavalle.pl/planning](https://lavalle.pl/planning/)
- **P2, vision backbones and transformers:** Stanford CS231n, Spring 2026. — [cs231n.stanford.edu](https://cs231n.stanford.edu/). Stanford CS336 *Language Modeling from Scratch*, Spring 2026, with YouTube recordings. — [cs336.stanford.edu](https://cs336.stanford.edu/)
- **P2/P3, flow matching and diffusion:** MIT 6.S184 (Holderrieth & Erives), 2026 edition. Labs culminate in building a latent diffusion model, and the notes are on arXiv. — [diffusion.csail.mit.edu/2026](https://diffusion.csail.mit.edu/2026/); [arXiv 2506.02070](https://arxiv.org/abs/2506.02070)
- **P3, RL:** Berkeley CS 185/285 (Levine), Spring 2026; the page links "Recordings of lectures from Fall 2023". — [rail.eecs.berkeley.edu/deeprlcourse](https://rail.eecs.berkeley.edu/deeprlcourse/). Book: Sutton & Barto, 2nd ed., free PDF. — [incompleteideas.net](http://incompleteideas.net/book/the-book-2nd.html)
- **P6, efficient inference and quantization:** MIT 6.5940 *TinyML and Efficient AI Computing* (Song Han), Fall 2026. — [hanlab.mit.edu/course](https://hanlab.mit.edu/course). Book: PMPP 5th ed., 2026, paid. — [Elsevier](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-443-43900-1)
- **P4:** the Tübingen group's lecture menu also lists "Deep Learning" and "Computer Vision" courses. Whether they cover NeRF/3DGS was not verified. — [Tübingen lectures menu](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/autonomous-vision/lectures/self-driving-cars/)

### Inferences
- A minimal "course spine" for Stage 0–3:
  - CS336 for transformers and KV caches (P2);
  - MIT 6.S184 for flow matching and SDEs (P2/P3);
  - CS285 for imitation learning, policy gradients and PPO (P3);
  - Tübingen SDC for the driving system view (P1).
- MIT 6.5940 pairs with Stage 4 (P6).

### Gaps
- CMU 16-831 (Introduction to Robot Learning): no official page found → UNVERIFIABLE.
- Course syllabi were not inspected topic by topic. Whether CS285 Spring 2026 covers DAgger, or 6.S184 covers ODE→SDE conversion, is unverified, though both are plausible from the course titles.
- No free course was located for IRT or measurement validity (P5) or for ISO/UL safety engineering (P7).

## Learning-method evidence supporting the mastery gates (including Anthropic arXiv 2601.20245)

### Takeaway
The gates in docs/03 §6 rest on robust findings:
- **Teach-back from memory** draws on the testing effect (Roediger & Karpicke 2006; rated "high utility" by Dunlosky et al. 2013) and on self-explanation (Chi et al. 1989).
- **Predict-before-measure and the transfer question** draw on the generation effect (d ≈ 0.40 across 86 studies).
- The Anthropic study exists: **arXiv 2601.20245, Shen & Tamkin 2026**. AI assistance cut quiz mastery by 17% (Cohen's d = 0.738) with no significant speed gain. Cognitively engaged interaction patterns preserved learning.

### Cited Findings
- **Testing effect.** "on the delayed tests, prior testing produced substantially greater retention than studying, even though repeated studying increased students' confidence in their ability to remember the material." Tests were taken after 5 minutes, 2 days or 1 week; restudy won only at 5 minutes. — [Roediger & Karpicke 2006 (PubMed)](https://pubmed.ncbi.nlm.nih.gov/16507066/)
- **Utility ratings.** Across 10 techniques, "Practice testing and distributed practice received high utility assessments because they benefit learners of different ages and a[bilities]…". — [Dunlosky et al. 2013 (PubMed)](https://pubmed.ncbi.nlm.nih.gov/26173288/)
- **Generation effect.** Across 445 effect sizes in 86 studies, the effect "was .40—a benefit of almost half a standard deviation of generation over reading". — [Bertsch et al. 2007 (PubMed)](https://pubmed.ncbi.nlm.nih.gov/17645161/). Original delineation: [Slamecka & Graf 1978](https://doi.org/10.1037/0278-7393.4.6.592)
- **Self-explanation.** "Good" students "generate many explanations which refine and expand the conditions for the action parts of the example solutions… guided by accurate monitoring of their own understanding", while "Poor" students "rely heavily on examples". — [Chi et al. 1989 (abstract via Crossref)](https://doi.org/10.1207/s15516709cog1302_1)
- **Retrieval vs elaboration.** Retrieval practice "produces greater gains in learning than elaborative studying with concept mapping" (Karpicke & Blunt 2011). This was disputed "on methodological and epistemological grounds" by Mintzes et al. — [Science comment (PubMed)](https://pubmed.ncbi.nlm.nih.gov/22034416/)
- **Anthropic study (arXiv 2601.20245).** Shen & Tamkin. The work was "done as a part of the Anthropic Fellows Program" (Shen); Tamkin is listed as Anthropic.
  - Design: randomized experiments on learning the Trio async library.
  - "AI use impairs conceptual understanding, code reading, and debugging abilities, without delivering significant efficiency gains on average."
  - Quiz effect: "a reduction in the evaluation score by 17% or two grade points (Cohen's d = 0.738, p = 0.010)", a 4.15-point gap on a 27-point quiz.
  - "We identify six distinct AI interaction patterns, three of which involve cognitive engagement and preserve learning outcomes".
  - The control group's gains are attributed to "encountering and subsequently resolving errors independently".

  — [arXiv 2601.20245](https://arxiv.org/abs/2601.20245); [HTML](https://arxiv.org/html/2601.20245)
- **Field evidence (high school).** GPT-4 access raised grades during practice (+48% GPT Base, +127% GPT Tutor). After access was removed, GPT Base students did 17% worse than students who never had access. The tutor's safeguards "largely mitigated" the harm. — [Bastani et al. 2025, PNAS (PubMed)](https://pubmed.ncbi.nlm.nih.gov/40560616/)
- **Worked examples and expertise reversal.** Records verified: [Sweller & Cooper 1985](https://doi.org/10.1207/s1532690xci0201_3) (title: worked examples "as a substitute for problem solving") and [Kalyuga et al. 2003](https://doi.org/10.1207/s15326985ep3801_4), "The Expertise Reversal Effect".
- **Learning by teaching.** Record verified: [Fiorella & Mayer 2013](https://doi.org/10.1016/j.cedpsych.2013.06.001), "The relative benefits of learning by teaching and teaching expectancy".

### Inferences
- docs/03 §6's "No stage is closed by a feeling of understanding" is directly supported by Roediger & Karpicke's finding that restudy *raised confidence* while lowering delayed retention.
- How the gates map onto the evidence:

  | Gate | Supporting evidence |
  |---|---|
  | Teach-back with a hand-worked number | Retrieval practice plus self-explanation |
  | Transfer question | Generation; transfer specifically is not separately verified |
  | Evidence gate | Not a learning-science claim; it is reproducibility |

- The Shen & Tamkin and Bastani results support PhysLoop's ownership rule: Huy writes the instruments, statistics, loss math and predictions, while agents write plumbing. Delegating those parts would likely reproduce the measured 17% mastery loss. The "three engaged patterns" suggest the AI-as-driver protocol in CLAUDE.md should demand Huy's prediction or derivation first, then AI comparison.

### Gaps
- Sweller & Cooper 1985, Kalyuga et al. 2003 and Fiorella & Mayer 2013: Crossref returned no abstracts, so effect sizes and conditions are not stated here.
- Health-literacy "teach-back" studies were not searched. Self-explanation and learning-by-teaching are used as the closest verified proxies.
- The precise names of the six interaction patterns in Shen & Tamkin (Figure 11) were not extracted.
