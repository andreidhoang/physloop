# Posting refresh for `05_ROLE_MAPPING.md` (2026-11-07)

This is an agent-authored evidence record for Dang Huy. It refreshes the posting facts in `05` §1 and supplies the evidence for the posting table in the Vietnamese explainer "PhysLoop giữa ngành" (https://claude.ai/artifact/4naU8Kz5bmhgNFPzuXt6Eo). It changes no stage, budget, gate or scientific decision. Posting lines are paraphrased, never quoted at length. Postings change, so re-verify a posting before applying to it.

## Verification levels

| Level | Items |
|---|---|
| Re-read against the full posting text by the coordinating agent | NVIDIA JR2010863, JR2022179, JR2019292 and JR2027161: requirement lines, years and printed pay bands |
| Re-queried by the coordinating agent through NVIDIA's public Workday API (title, location, `startDate`, `canApply`, printed pay) | JR2025038, JR2028518, JR2024273, JR2016777, JR2016768, JR2014137, JR2023529, JR2028334 and JR2029093 are live; JR2019461 is not listed |
| Collected by a research subagent through public ATS APIs, not re-read line by line | Everything else in §§1–4 below |

The table below gives the coordinating agent's Workday re-query of IDs that were in the 2026-10-27 table but outside the subagent's priority list.

| ID | Title as listed | `startDate` | Printed base pay (USD) |
|---|---|---|---|
| JR2025038 | Senior Integration Engineer, End-to-End Model – AV | 2026-11-05 | L3 152,000–241,500; L4 184,000–287,500 |
| JR2028518 | Senior Algorithm Engineer – AV | 2026-10-30 | L4 184,000–287,500; L5 224,000–356,500 |
| JR2024273 | Senior Software Engineer, AV Planner – AV | 2026-10-30 | 224,000–356,500 |
| JR2016777 | Senior Software Performance Engineer – AV | 2026-10-30 | L5 224,000–356,500; L6 272,000–431,250 |
| JR2016768 | Manager, Deep Learning – AV and Robotics | 2026-10-30 | L3 224,000–356,500; L4 272,000–431,250 |
| JR2014137 | Lead Safety Architect – AV | 2026-10-30 | 224,000–356,500 |
| JR2023529 | Director, Engineering – AV | 2026-10-30 | 320,000–488,750 |
| JR2019461 | (Senior Research Manager, World Model Evaluation) | — | Not listed; still not live |

**Not done in this refresh:**
- The full sweep of NVIDIA's AV postings was not repeated, so the 2026-10-27 count of 77 AV-titled engineering postings was not re-counted.
- The skill-demand percentages in `05` §2 were not recomputed.

## Subagent ledger (verbatim, 2026-11-07)

**Method.** NVIDIA: Workday JSON API (search + `/job/<path>` detail), read 2026-11-07. Waymo, Nuro, Motional, Kodiak, Torc and XPeng: Greenhouse boards API. Zoox and Waabi: Lever API. Wayve, Applied Intuition and Physical Intelligence: Ashby API. A posting counts as "Live" when it is in the API listing today; NVIDIA postings also return `canApply=true`. Pay is the base range as printed. I did not apply, sign in or submit anything.

**NVIDIA caveat.** Every Santa Clara NVIDIA posting below says applications are accepted "at least until November 3, 2026". That date has passed. The postings are still live today, but they can close at any time. The posted date is Workday's `startDate`, which is a mass refresh on 2026-10-30. NVIDIA prints "eligible for equity and benefits" and mentions no bonus.

### 1. NVIDIA (priority IDs re-verified)

| ID | Title / location | Posted | Live? | Base pay as printed | Key requirements (paraphrased) |
|---|---|---|---|---|---|
| **JR2010863** | Senior ML and Simulation Engineer – AV (AV Simulation team), Santa Clara | 2026-10-30 | Live | **L5 $224,000–$356,500; L6 $272,000–$431,250** + equity | BS or equivalent; **12+ yrs** across large-scale ML training, AV, simulation and AI infra. Deep PPO/GRPO with hyperparameter and **reward-function design**. C++/Python; GPU clusters with Slurm or K8s. Plus: RL or LLM post-training infra, and **closed-loop eval of E2E AV models**. Role builds a closed-loop sim RL framework for Alpamayo-R1-class models (NuRec, traffic models, Cosmos) and measures **simulation quality** |
| JR2022179 | Senior Deep RL Engineer – Autonomous Driving, Santa Clara | 2026-10-30 | Live | $224,000–$356,500 + equity | BS+; **12+ yrs**. Policy gradient (PPO/GRPO), actor-critic, on- and off-policy RL. PyTorch; C++ for real-time systems. Benchmarks RL against IL baselines. Plus: flow-matching/diffusion policies, training on own rollouts (compounding error), data flywheels |
| JR2005597 | Senior Research Engineer – AV (AV Research), Santa Clara | 2026-10-30 | Live | L4 $184,000–$287,500; L5 $224,000–$356,500 | BS+; **10+ yrs** MLOps/AI infra. Distributed training. PPO/SAC with reward tuning, reward shaping, domain randomization and curriculum. CUDA; Slurm/K8s. Builds GPU sim infra for policy training and RL fine-tuning of multimodal LLMs |
| JR2005596 | Senior Research Scientist – AV, Santa Clara | 2026-10-30 | Live | L4 $192,000–$304,750; L5 $224,000–$356,500 | **PhD** (or equivalent) plus 6+ yrs autonomy research. Top-venue publications (RSS/ICRA/NeurIPS/CVPR). Python, C++, CUDA. Topics include closed-loop training |
| JR2014651 | Senior SWE, Metrics and Evaluation – AV, Santa Clara + US remote (GA/DC/NC/WA) | 2026-10-30 | Live | L4 $184,000–$287,500; L5 $224,000–$356,500 | MS/PhD or equivalent; **8+ yrs** SWE, data science or experiment design. Python stack. Plus: AV metrics, precision/recall of metrics, C++, Bazel |
| JR2020021 | Listed as "Systems Engineer, World Model Evaluation – AV"; URL slug still reads "Senior Software Engineer, World Model Systems Engineering" (retitled), Santa Clara | 2026-10-30 | Live | L4 $184,000–$287,500; L5 $224,000–$356,500 | MS/PhD; **8+ yrs** V&V, experiment design or quantitative analysis. Evaluators linking model error to vehicle behavior. Statistics, **measurement uncertainty**. Python/SQL pipelines. Plus: closed-loop linkage, Temporal, validating VLM/LLM outputs |
| JR2027161 | Deep Learning Senior Engineer, E2E Autonomous Driving, Santa Clara / CA remote | 2026-10-30 | Live | L4 $184,000–$287,500; L5 $224,000–$356,500 | PhD+4 or MS+6 yrs. Build LLM/VLM/VLA systems, or a top-tier coder record. Generative, IL and RL for planning. Production AV/robotics ML. C++ deployment |
| JR2019292 | Senior SWE – AD Simulation, Santa Clara | 2026-10-30 | Live | **L3 $152,000–$241,500; L4 $184,000–$287,500** | BS/MS/PhD; **5+ yrs AV industry**. Real2Sim/Sim2Real. Python, C++, PyTorch; Docker, Bazel, CI. Plus: first-author papers, neural rendering, open source |
| *New:* JR2029093 | Principal Engineer – AV Simulation and Evaluation, Santa Clara / US remote | 2026-11-02 | Live (window "until Nov 6", now passed) | $272,000–$431,250 | 15+ yrs distributed and data-intensive systems. Eval infra at hundreds of millions of runs per day. Arrow, Parquet, MCAP; Temporal |
| *New:* JR2028334 | Senior SWE, AV Analysis Platform, Santa Clara / US remote | 2026-10-30 | Live | $224,000–$356,500 | Requirements not read in detail |
| *New:* JR2026501 | Technical Lead, GenAI – AV, Santa Clara | 2026-10-30 | Live | $224,000–$356,500 | Cosmos/Omniverse platform lead (adjacent role) |
| *New:* JR2028022 | RL and World Model for Autonomous Simulation **Intern** 2027, Shanghai | 2026-10-23 | Live | Not printed | PhD student. Policy gradient, actor-critic, offline RL. Deploys world-model policies into simulated RL environments |
| *New:* JR2019758 | DL Senior Engineer, E2E AD, Shanghai / Beijing | 2026-08-29 | Live | Not printed | Same description as JR2027161, but lower bar: PhD+2 or MS+4 yrs |
| *New:* **JR2029271** | Senior Research Scientist – AV, **Singapore** (Suntec) | 2026-11-03 | Live | Not printed | Same requirements as JR2005596: PhD, 6+ yrs, top-venue publications |
| JR2024056 | DL Solution Architect – RL and Post-training, Beijing | 30+ days | Live | Not printed | Adjacent role; not read in detail |

**Other NVIDIA Asia.** NVIDIA Vietnam has no AV, RL or simulation role open. The closest roles in Ho Chi Minh City are not AV and print no pay:
- JR2001271, Senior DL Algorithm Engineer (perf optimization; 5+ yrs, MSc/PhD), posted 2026-01-11;
- JR2028594, Senior System Software Engineer (automation infra; 6+ yrs), posted 2026-10-27.

Japan has only JR2015317, Principal Engineer, AV and Physical AI Solutions (a solutions role). I found no AV roles in Taiwan.

### 2. Other AV companies

| Company | Title (ID) | Location | Posted / updated | Live? | Base pay as printed | Key requirements (paraphrased) |
|---|---|---|---|---|---|---|
| Waymo | **Research Scientist, RL for Autonomous Planning and World Modeling** (gh 8165872) | MTV/SF/Kirkland/NYC | pub 2026-08-29, upd 10-20 | Live | **$204,000–$259,000** + discretionary bonus + equity | PhD/MS plus 3+ yrs industry or postdoc in RL or foundation models. Post-training and evaluation of Waymo's foundation world model. RL plus distillation for trajectory planning; ablations on internal RL infra |
| Waymo | Senior MLE, Simulation (7430187) | MTV/NYC | pub 2025-12-08 (refreshed 10-22) | Live | $204,000–$259,000 + bonus + equity | 2+ yrs productionized ML (4+ preferred). RL, transfer or related learning. Driver Understanding and Evaluation team |
| Waymo | Staff MLE, Simulation (8056720) | MTV/NYC | pub 2026-07-13 | Live | $251,000–$310,000 + bonus + equity | 5+ yrs (8+ preferred) productionized ML; RL experience |
| Waymo | Senior SWE, Quantitative Evaluations (7477172) | MTV/SF | pub 2025-12-18 | Live | $204,000–$259,000 | Planner evaluation. Build and **validate metrics**; link onboard changes to simulated outcomes; 2–3 yrs+ |
| Waymo | Staff TLM, ML, Simulator Evaluation (8209790) | MTV/SF | pub 2026-09-17 | Live | $251,000–$310,000 | 7+ yrs production ML plus 3+ yrs managing. Eval frameworks for world-model and **simulator quality**. MS/PhD |
| Waymo | Interns (2027): Simulation and Closed-Loop Training PhD (8283558); Behavior Post-training PhD (8289816); Simulator Realism Eval (8214350) | MTV/SF | Sep–Oct 2026 | Live | Not extracted | PhD students only |
| Wayve | **Staff/Senior MLE, Reinforcement Learning** (ashby 5df6d6f2) | Sunnyvale | 2026-10-05 | Live | **Not printed** (comp fields empty) | RL beyond behaviour cloning for E2E driving: offline/off-policy RL, **reward models**. Evidence across open-loop, **closed-loop sim** and on-road; avoid benchmark overfitting. Plus: off-policy evaluation, calibration |
| Wayve | Software Engineer, Simulation (90866bb9) | Sunnyvale (also London) | 2026-10-14 | Live | $209K–$310K + equity | C++/Python simulator. Root-causes on-road vs simulation gaps; sim reproducibility and performance |
| Applied Intuition | **Research Engineer – RL, Self-Driving** (ashby 36a49d01) | Sunnyvale | 2026-02-13 | Live | **$126K–$423K** (Ashby band, all levels) | MS/PhD. Large-scale self-play RL, **VLA post-training**, closed-loop RL on neural simulation; open to all experience levels |
| Applied Intuition | Research Engineer – AI/RL Infrastructure (091ed808) | Sunnyvale | 2024-09-17 (still listed) | Live | $126K–$423K | Training/eval infra; continuous eval and regression tracking; Senior/Staff preferred |
| Applied Intuition | SWE, SDS Core – Evaluation (e015ce66) | Sunnyvale | 2026-02-10 | Live | $125K–$222K | 3–5 yrs. Own autonomy metrics and datasets, real and synthetic |
| Nuro | Staff/Lead MLE, Behavior and Planning (gh 8114754) | Mountain View | pub 2026-08-07 | Live | $235,030–$352,290 | 7+ yrs. World models, RL/IL, diffusion planning; production impact over papers |
| Nuro | Senior SWE, Autonomy Behavior Testing (8282222) | Mountain View | 2026-10-14 | Live | $160,360–$291,150 | 5+ yrs (3+ in AV testing). Large-scale sim and test pipelines |
| Zoox | Senior Staff/Principal MLE – Autonomy Behaviors (lever b30b28b0) | Foster City | 2025-11-04 | Live | $340,000–$492,000 | 10+ yrs. IL, RL and scaling for the ML planner; metrics and training frameworks |
| Zoox | Manager – Model Evaluation (8714ca37) | Foster City | 2026-09-14 | Live | $237,000–$338,000 | MS/PhD; 7+ yrs technical plus 3+ yrs managing. Open- and closed-loop validation, **statistical release gating** |
| Waabi | Research Scientist, Simulation Agents (lever 48a13158) | Remote US/CA | 2026-03-25 | Live | US $158,000–$269,000 + equity | MS/PhD. IL/RL for sim agents; traffic simulation |
| Waabi | Senior SWE, Evaluation Infrastructure (12c40987) | Toronto / SF / remote | 2026-03-26 | Live | US $127,000–$223,000 | 2+ yrs. Metrics and tags over real and sim data |
| Motional | Senior Engineer, AV Behavior Evaluation (gh 7767449003) | Pittsburgh / remote US | pub 2026-06-09 | Live | $149,000–$198,500 | BS+2 yrs. Metric standardization; **statistical metric targets**; sim/resim analysis |
| Kodiak | Staff SWE, Learned and Hybrid Behavior Planning (4294898009) | Mountain View | upd 2026-11-02 | Live | $240,000–$265,000 (across levels; equity noted) | Model evaluation and debugging via sim, logs and metrics |
| XPeng (US) | Senior Staff MLE – AD Foundation Models (gh 8465375002) | Santa Clara | pub 2026-03-16 | Live | $244,140–$413,160 + bonus + equity | 5–8 yrs DL. World models for the long tail |
| Tesla | AI Engineer, RL, Self-Driving | Palo Alto | Aggregator: posted 2026-05-05 | **Expired** (Career.com: closed 2026-07-28) | Aggregator only: $176K–$420K (cash + stock), **unverified** | Extensive RL R&D; Python; PyTorch/JAX. tesla.com returned 403 to the API and WebFetch, so I could not check whether any Tesla posting is live |

**Could not locate or verify:**
- **Aurora:** the careers page loads its listings dynamically, so I found no board. Aggregators show roles such as ML Eng TL, Behavior Planning ($171K–$247K, March 2026), all unverified.
- **Torc:** only simulation V&V and test roles.
- **Physical Intelligence:** no RL role with a printed salary.
- **Google DeepMind:** not checked (no AV relevance found).
- **Momenta and Li Auto:** no postings found. Momenta announced its R7 "RL world model" in April 2026.
- **Grab:** not found.
- **VinMotion / VinRobotics:** the VinMotion RL page shows a maintenance notice and the VinRobotics RL page returns 404. Aggregators show Hanoi RL and humanoid roles (2025–26) with no salary.
- **Motional Singapore:** Senior/Staff SWE Motion Planning (7798656003) and an ML internship, Behaviors Research (7600024003), both live, no pay printed.

### 3. Compensation references (self-reported aggregates, not offers)

- **levels.fyi, NVIDIA Machine Learning Engineer, US** (n=36, updated 2026-10-01). Averages:
  - IC2: $211K total ($165K base);
  - IC3 "Senior": $315K ($203K base, $96K stock);
  - IC4: $340K ($231K base);
  - median: $242K.
- **levels.fyi, NVIDIA Software Engineer, US** (n=2,376, updated 2026-11-07). Median total comp by level:
  - IC3 "Senior": $319K;
  - IC4: $359K;
  - IC5: $537K;
  - IC6: $650K.
  - **Mapping caveat:** levels.fyi's IC numbers may not equal the "Level 3–6" that NVIDIA postings print. I have not verified the mapping.
- **levels.fyi, Waymo MLE, US** (n=14, thin sample; updated 2026-10-03). Averages:
  - L4: $335K ($204K base);
  - L5: $494K ($238K base, $213K stock);
  - median: $372K.

### 4. Summary

**Common skills across postings:**
1. RL for driving policies beyond behaviour cloning:
   - PPO/GRPO, offline and off-policy RL;
   - reward design and reward models;
   - compounding error and training on the policy's own rollouts.
2. Closed-loop simulation as both training environment and evaluator, including measuring simulator realism and quality (NVIDIA JR2010863, Waymo Sim Eval, Wayve).
3. Metric engineering with statistical rigor:
   - validating the metrics themselves;
   - uncertainty;
   - release gating;
   - separating real gains from benchmark overfitting.
4. Large-GPU-cluster training and eval infrastructure (Slurm/K8s, distributed PyTorch). Python plus C++ is near-universal.
5. Pluses: VLA/LLM post-training, diffusion or flow-matching policies, world models, publications or open source.

**Experience bar.** The primary target, **JR2010863**, is a staff-level req that asks for **12+ years**. NVIDIA's other RL and infra reqs ask for 10–15+ years (JR2022179 12+, JR2005597 10+, JR2029093 15+), and the eval reqs ask for 8+ (JR2014651, JR2020021). The most reachable NVIDIA AV reqs are:
- **JR2019292** (5+ yrs AV industry, L3/L4);
- **JR2027161** (MS+6 or PhD+4);
- **JR2019758** in Shanghai (MS+4 or PhD+2).

Research-scientist reqs require a PhD and top-venue publications. Outside NVIDIA, Applied Intuition RL (open to all levels, MS/PhD), Waymo senior roles (2–4+ yrs) and Waabi/Motional eval roles (2+ yrs) have lower bars.

**Posted US base bands by level:**
- NVIDIA: L3 $152–242K, L4 $184–288K (research $192–305K), L5 $224–357K, L6 / Principal $272–431K, plus equity.
- Waymo: Senior $204–259K, Staff $251–310K, plus bonus and equity.
- Others:
  - mid/senior eval: $125–260K;
  - staff ML: $235–413K;
  - Zoox principal: $340–492K.

**Asia.** None of the Asian postings print pay. Relevant live roles:
- NVIDIA **Singapore** JR2029271 (PhD research);
- NVIDIA Shanghai/Beijing E2E (JR2019758);
- the NVIDIA Shanghai RL intern role (JR2028022).

Nothing AV-related is open at NVIDIA Vietnam.
