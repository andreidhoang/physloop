# S2 CPU handoff — engineering acceptance

Agent-authored, before implementation. Scope: plan S2.1/S2.2/S2.3/S2.7 [C],
G1/G2/G3·R1/G7, under D-017/D-018 and the owner's CPU implementation instruction.
Acceptance contract: `docs/S2_CPU_HANDOFF_SPEC.md`, E0–E4 and its seven literal
positive/negative cases. These are deterministic engineering checks, not a
scientific experiment or Huy's prediction.

Expected outputs: a pinned actual-format ASL parser, receipt/attempt inspection,
source-checked five-scene command preparation, updated setup facts, and a fresh
verification bundle including the clean-rebuild result or exact blocker.
Expected cloud/GPU cost: $0. Existing source/evidence and protected inference
remain unchanged; no model, gated dataset, paid run, license action or commit.

Execution results will be appended after the frozen-source verification.

## Implementation and review record before the final run

Tests first failed on the missing ASL/capture/launch modules, then on the missing
CLI commands and missing rehearsal outputs. Independent source review required
retaining all request receipts, validating decoded identity semantics rather than
syntax alone, bounding file size before reading, and exercising the actual wire
decoder through the capture wrapper. Those changes are implemented, including a
literal-wire independent regression with 62 rejected torn suffixes.

Launch review found that JSON round-tripping is not general Hydra escaping, and
the pinned wizard later joins command strings for a container shell. The supported
v0 input alphabet is now explicit and narrower. Source review also required
`external_image=true` for physics/controller/runtime when using immutable prebuilt
image references, and the existing pilot spec's renderer timing flag in both arms.
These fixes have test-first regressions and source-derived fixture checks.

Independent reviewers found no remaining blocker within the bounded parser,
capture-inspection and command-preparation contracts. Actual simulator/scorer
qualification remains outside those CPU results. Public fixture provenance
includes exact source bytes, SHA-256 and Git blob identity checks.

Environment attempts are retained in `evidence/s2-environment-20261107-v1/` and
`-v2/`. A fresh virtual environment was created, but exact locked package downloads
failed DNS; uv 0.8.17 also panics in macOS SystemConfiguration, and transfer of an
official newer uv artifact failed. Protobuf 7.36.2 was installed in the working
environment from a RECORD-verified cache and its exact previously resolved lock
entry retained with provenance. This is not fresh resolution or a clean rebuild.
The final run will verify current installed versions separately and preserve
`clean_frozen_sync_verified=false`.
