# PhysLoop

PhysLoop builds and measures NVIDIA's Alpamayo **teacher → Thor-student** path for autonomous driving. It has two parts:
- **the gate**, an open release gate that runs paired, stratified, validity-checked closed-loop verdicts on AlpaSim;
- **PhysLoop-AV**, the loop the gate measures:
  - NVIDIA's own 2B student against its 10B teacher;
  - closed-loop RL of the teacher;
  - on-policy correction of the student and its reasoning-budget curve;
  - real-time serving on Jetson AGX Thor with hardware-in-the-loop;
  - optional shadow mode in Ho Chi Minh City.

The gate answers one question about the model that actually ships:

> Did distillation, quantization or RL make the deployable model worse, and where?

**Why it is needed** (verified 2026-10-25):
- NVIDIA has announced Alpamayo 2 Edge, a 2B model for DRIVE AGX Thor at 10 Hz. It has also put up a public 10B → 2B distillation recipe (alpamayo-recipes PR #42, unmerged). Both report open-loop error and latency only. Nobody has published how the student drives in closed loop.
- NVIDIA's own 2B students run at 171 ms (FP8) on DRIVE AGX Thor, so what it takes to reach 10 Hz is an open, measurable question.
- No open tool gives a paired, stratified go/no-go verdict between two Alpamayo checkpoints. NVIDIA's Alpamayo 1.5 card moved its AlpaSim score from 0.81 to 1.37 with unchanged weights.
- AlpaGym has published no closed-loop RL gains.

## The loop (systems view)

```
        ┌──────────────────────────────────────────────────────────┐
        │                REINFORCING LOOP (data flywheel)           │
        ▼                                                          │
 [fleet-style logs] → [NuRec reconstruction] → [AlpaSim closed-loop]│
                                                   │               │
                       [Teacher: Alpamayo 1.5 10B / 2 Super ~34B]  │
                                   │ distill / RL / quantize       │
                                   ▼                               │
                     [Student 2B (Edge), FP8/NVFP4, Thor]          │
                                   ▼                               │
                    ╔════════════════════════════╗                 │
                    ║  RELEASE GATE (this repo)   ║ ◄── BALANCING   │
                    ║  paired + stratified +      ║     LOOP        │
                    ║  validity + reliability     ║                 │
                    ╚════════════════════════════╝                 │
                                   │ go                            │
                                   ▼                               │
          [Jetson AGX Thor → HIL → shadow mode (open-loop)] ───────┘
```

The gate is the balancing loop. Without it, the flywheel optimizes whatever the metric rewards (Goodhart), and RL is the strongest metric-exploiter we own.

## Status
- Standalone and active. Plan v4 dated 2026-10-25 (DECISIONS D-012).
- Stage S1: the gate decision core on synthetic truth plus a power analysis. It costs $0 and runs on CPU; D = 2026-10-26.
- No implementation exists yet.

## Start here
1. `DOC_AUTHORITY.md` — which document wins.
2. `docs/00_START_HERE.md` — reading order, the build rituals, and where we are.
3. `docs/02_PHYSLOOP_PLAN.md` — what we build, in what order, and what kills it.

When working with Claude Code, run `/reread` before every build session.
