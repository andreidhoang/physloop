# Completeness audit of the D-024 CPU build (before the final round)

Independent agent audit, 2026-11-09, read-only. Recorded verbatim from the auditor's structured result before the final fix round, so later changes can be diffed against it. Agent-authored; not a scientific registration, admission or runtime qualification.

**Suite at audit time:** PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -p no:cacheprovider -q tests (2026-11-09T00:07Z): 4547 passed, 1 skipped in 132.21 s. The skip is tests/test_s2_guest_manifest.py:136 ("filesystem normalizes names"). torch 2.8.0 and safetensors 0.7.0 are installed in .venv, so tests/s4_torch ran (88 test functions). I ran the suite a second time with -rs only to find the skip reason, and the counts were the same. ruff check --no-cache on gate, teacher_rl, scripts and tests is clean; git diff --check is clean; gate/verdict.py, gate/stats/ and .claude/ are unchanged. These rehearsals passed in the scratch dir: scripts/rehearse_s2.py --fixture complete (10 units, 11 attempts, 2,646 sweep units, 104 bundle entries, 12 replayed analyses); scripts/rehearse_s4.py; scripts/rehearse_s2_pilot.py --demo-only.

**Auditor summary:** The D-024 CPU build is broad and well tested: 4,547 passed and 1 skipped (S4 torch tests included). Ruff and whitespace checks are clean, protected paths are untouched, and both synthetic rehearsals pass. I re-checked the E19 and E20 known answers independently. It is not ready for renting GPUs.  Three engineering gaps block real use: - No extraction from real AlpaSim output, so actual pilot or full data cannot become observations, validity reports or gate records. - The S4 capture's trainer evidence has no consumers, so most E-checks cannot run on a real training run. - The guest runner is never provisioned. It needs Python ≥3.11 and its dependencies; Ubuntu 22.04's python3 is 3.10.  Safety issues: - The real Vast `step` command accepts any decision ID in DECISIONS.md, including closed D-019 and no-rental D-024. - The S4 launcher and the unit runner accept free-text authority, and the runner has no preflight gate. - The S4 watchdog force-removes containers by name pattern. - Destroy follows a typed export hash, not a verified export.  Medium gaps: the per-unit budget is not fed by the cost ledger; the report trusts declared evidence; S4.6 guards are not registered or bound; superiority sensitivity is not integrated; telemetry producers are missing. No final frozen rebuild covers the current lock. 32 owner inputs remain open.

## Coverage

| Ticket | Modules | Level | Status | Note |
|---|---|---|---|---|
| CPU-00 | scripts/verify_s2.py, scripts/verify_s2_ingestion.py (frozen snapshot), gate/s2_bundle.py (source manifest, source_drift), scripts/rehearse_s2.py | L1 | partial | tests/test_s2_handoff.py is absent. No consumer rejects an artifact that lacks a producer identity: s2_report evidence rows have no producer field. The rehearse_s2 SOURCE_GLOBS (:90) leave out teacher_rl/ and git dirty state. Protocol-hash binding works through schedule and packet rebuilds. |
| CPU-01 | gate/s2_runtime.py validate_runtime_inventory; experiments/S2_SETUP_DRAFT.json runtime_inventory; tests/test_s2_runtime.py | L1/L2 | covered | The setup draft validates structurally and is blocked on authorization_missing, owner_input_missing, disk_capacity_unset, host_path_unset, image_reference_unset, local_build_image_id_pending and toolchain_unset. Unknown sizes are never zero. Real bytes and platform manifests are L3. |
| CPU-02 | gate/s2_readiness.py inspect_readiness; tests/test_s2_readiness.py (25) | L1 | covered | Separate sections; it rejects closed D-019, synthetic evidence in an actual plan, and changed bytes. No executing CLI (s2_lifecycle, s2_run_units, host_launch) calls it. |
| CPU-03 | gate/s2_population.py audit_population; tests/test_s2_population.py; tests/fixtures/s2_analysis | L1 | covered | Every literal answer is tested: [0,10)∩[8,12)=2, the half-open boundary gives 0, 3 scenes on 2 roads, road with two strata rejected, changed asset rejected, reorder invariant, alpamayo_aligned treated as a label. overlap_rule None gives blocked. Nothing builds the manifest from the catalog or licensed metadata (data-gated). |
| CPU-04 | gate/s2_diagnostics.py describe_validity; tests/test_s2_diagnostics.py | L1 | covered | Bins, 2×2 table with unknowns outside it, retry counts, the late 4.2 m case, black-frame plus NaN pose, and rejected-batch diagnostics are all tested over the synthetic W3 producer. Corridor and overshoot stay unknown. No actual W3 producer exists (see gaps). |
| CPU-05 | gate/s2_probes.py describe_repeats/compare_probe_artifacts; gate/tensor_manifest.py; tests | L1 (+L3 probes) | covered | Tested: [0,1,0] sample variance 1/3 and [1,1,1] variance 0; zero discordance gives not_estimable; method None gives method_unregistered; QUALIFIED_METHODS is empty, so any method gives unsupported_method; same bytes with a different dtype or shape is not parity. There are no host probe runners. |
| CPU-06 | gate/s2_power.py run_sensitivity/zero_difference_minimum_roads; tests/test_s2_power.py | L1 | covered | Tested: 36/51/72/183 roads with each previous integer failing; upper bound 0.04364750021196295; sequential looks through SequentialGate; exact binary enumeration oracle; irregular layouts. Non-inferiority only: superiority is not integrated (see S2.6). |
| CPU-07 | gate/s2_telemetry.py summarize_telemetry; scripts/s2_gpu_sampler.py; tests | L1 (L2 names, L3 delivery) | partial | Tested: 1 GPU-hour, 8 episodes per wall-hour, 4 per GPU-hour, 1000 J, gap gives incomplete, counter reset, NO_DATA is not zero XIDs. Missing: a DCGM .prom parser, and any producer of the allocation segment partition or the useful-flagged attempt rows from the journal or ledger. The stabilization rule is the owner's. |
| CPU-08 | gate/s2_cost.py reconcile_cost/can_start_next_unit; scripts/s2_lifecycle.py reconcile-cost/next-unit; tests/test_s2_cost.py | L1 | covered | Tested: $10 deposit plus $3 usage gives $7 credit; 61 s bills as 2 min; strict equality boundary; D-020's $0.60. Not wired to the per-unit check in scripts/s2_run_units.py (see gaps). |
| CPU-09 | gate/s2_bootstrap.py build_execution_plan/validate_plan/evaluate_rule; scripts/s2_guest_bootstrap.py; scripts/vast_s2_host_probe.sh; teacher_rl/ops.py | L1 plan / L2 argv / L3 exec | partial | Phases A–E compile all-or-nothing behind terms:/acquire: refs; secrets come from files; resume works by receipts; probe pulls are digest-only (:134). No relocatable bundle (scripts/s2_prepare_bundle.py is absent). No stage installs or verifies the PhysLoop runner itself (needs Python ≥3.11 plus dependencies). |
| CPU-10 | gate/s2_execution.py (validate_action_record, LifecycleJournal, next_action, Coordinator, VastCliClient, FakeProvider); scripts/s2_lifecycle.py; scripts/vast_stop_watchdog.py | L1 fake / L2 CLI / L3 provider | covered | 77 transition tests: uncertain create, crash and restart, duplicate label, D-020 exited shape, 12 h window, two consecutive NotFound. Vast only. Weaknesses: the real-provider authority check, and destroy following a typed export_verified (see safety). |
| CPU-11 | gate/s2_bundle.py seal/verify/replay; scripts/s2_guest_manifest.py; gate/s2_pilot.verify_export/copy_verified_export | L1 | covered | Self-excluding manifest, final seal marker, secret canaries, restriction classes and fresh-directory replay all work. dependency_installation_tested is false. The lifecycle never consumes a verification report. |
| CPU-12 | tests/test_s2_evaluator_path.py; tests/fixtures/alpasim_s2_raw (hash-checked e121e59, Apache-2.0) | L2 | covered | Warm-up removal, first-incident truncation, missing-offroad zero-fill (normalization stays blocked), invalid frames, the ≥4 m cut and the scorer call site are tested. Only AST-extracted helpers run; the whole evaluator and scene_score stay not_run by design. |
| CPU-13 | scripts/rehearse_s2.py; gate/s2_report.py; gate/s2.py CLI; tests/test_s2_rehearsal.py, tests/test_s2_report.py | L1 | partial | The rehearsal passes with controls and replay. The report trusts declared evidence rows. The suite, Ruff and whitespace checks are not run inside the sealed snapshot. The final clean frozen rebuild for the current lock has not been produced. |
| E01 | teacher_rl/capture/entrypoint.py (session_started random_seed); teacher_rl/records.py randomness_record; teacher_rl/parity.py compare_trajectories | L1/L3 | partial | Seeds are captured, but no check compares two captured runs that have identical weights, inputs and noise. |
| E02 | teacher_rl/export_audit.py audit_export/check_zero_update_export; teacher_rl/parity.py (cross-dtype ladder); scripts/s4_parity_check.py; capture checkpoint_save | L1/L2 (L3 tensors) | covered |  |
| E03 | teacher_rl/params.py snapshot/diff/check_trainable_set; scripts/s4_snapshot_params.py; tests/s4_torch/test_s4_params.py | L1/L2 | covered | Covered on checkpoints. The in-process parameter_snapshot capture has no adapter into s4_param_snapshot (see gaps). |
| E04 | tests/s4_torch/test_s4_replay_objective_upstream.py (joint vs geometric-mean ratio); teacher_rl/parity.py; capture minibatch_started.old_logprobs and replay_forward.log_probs | L1/L2 | partial | No check pairs old and new log-density at equal behaviour version, before the first optimizer step, at μ=1. |
| E05 | teacher_rl/checks/audit.py check_version_fence; records.trace_versions; capture generation_entered/exited | L1/L2 | covered | Offline rejection; allowed lag is the owner's input. The dispatcher path is L3. |
| E06 | teacher_rl/records.py build_attempt_ledger; checks/signal.py check_advantage_recomputation | L1 | covered |  |
| E07 | checks/signal.py check_equal_reward_control (criterion required); checks/steps.py check_skip_semantics; teacher_rl/adapters.py; pinned skip-path torch test | L1/L2 | covered | G=6 tied groups give nonzero advantages (about 28% in my check). The capture records optimizer_step_calls and grad_norm but no optimizer-moment digests. |
| E08 | checks/signal.py check_advantage_ordering | L1 | covered |  |
| E09 | teacher_rl/records.py check_trace_integrity | L1 | covered |  |
| E10 | checks/semantics.py reconcile_metric_domains | L1/L2 | covered | Real inputs need a decoder for captured simulate_call replies, which does not exist. |
| E11 | checks/semantics.py contact_predicates | L1 | covered |  |
| E12 | checks/semantics.py check_reference_alignment/reference_rmse | L1 | covered |  |
| E13 | checks/semantics.py diagnose_trajectory (thresholds required) | L1 | covered |  |
| E14 | checks/semantics.py summarize_validity; teacher_rl/diagnostics.py | L1 | covered |  |
| E15 | teacher_rl/records.py check_run_identity; capture/host_launch.py tree verification | L1/L2 | covered |  |
| E16 | checks/audit.py audit_adapter_paths; adapters.audit_train_gate_paths | L2 | covered |  |
| E17 | checks/audit.py uncertainty_axes; records.describe_run_randomness | L1 | covered |  |
| E18 | teacher_rl/watchdog.py; gate/s2_execution.py hand-off; teacher_rl/forecast.py | L1 (L3 host) | partial | No operator entry point and no watch-spec producer. Containers are removed by name pattern. |
| E19 | checks/signal.py group_advantages_f32 | L1 | covered | Re-verified: [0,0.001] gives ±0.998004; [0,10] gives ±0.9999998; G=6 with one collision gives 5×0.4472135 and −2.2360673; tied G=2/4/8/16 give exactly 0. |
| E20 | checks/steps.py expected_optimizer_steps/reconcile_optimizer_steps; pinned-loop torch test | L1/L2 | covered | Re-verified: (44,1,1,0) gives 44; (10,4,2,1) gives 5. |
| E21 | checks/audit.py kl_status/simulate_retention; capture export hashing before pruning | L1/L2 | covered | kl_beta 0 gives not_computed. |
| S2.1 | s2_runtime, s2_bootstrap, s2_preflight, s2_readiness, S2_SETUP_DRAFT.json | L1/L2 | owner_blocked | Pilot preflight has 24 blockers and full preflight 37. Registration and rights are the owner's. |
| S2.2 | s2_packet, s2_launch, s2_executor, scripts/s2_run_units.py, s2_execution, s2_cost | L1 (L3 run) | owner_blocked | Engineering is present. Caveats: the runner's budget check is not ledger-fed, and the runner has no preflight or readiness gate. |
| S2.3 | s2_extract (synthetic), s2_ledger v2, s2_capture, s2_asl, s2_actual, test_s2_evaluator_path | L1/L2 | partial | No actual-AlpaSim extraction. |
| S2.4 | s2_probes, tensor_manifest, full-sweep A/B packet | L1 | partial | Reliability method is unregistered. Probe runners and the re-export procedure are missing. |
| S2.5 | s2_diagnostics, s2_population | L1 | partial | The logic is done; producers for actual data are missing. |
| S2.6 | s2_power, superiority_power, s2_gate | L1 | partial | Non-inferiority only in run_sensitivity; superiority is not integrated; only fixed_sample works in the registered glue. |
| S2.7 | s2_telemetry, s2_gpu_sampler, s2_cost | L1/L2 | partial | Telemetry producers are missing; the stabilization rule is the owner's. |
| S4.1 | teacher_rl/registration.py, pins.py, launch.py, experiments/S4_SETUP_DRAFT.json | L1/L2 | owner_blocked | Every slot is required with no default; there is no slot for the AlpaGym issue #10 response. |
| S4.2 | params, parity, export_audit, checks/signal, checks/steps, adapters | L1/L2 | partial | E04 composition and the capture-to-check adapters are missing. |
| S4.3 | capture/entrypoint, sink, host_launch; checks/semantics; records | L1/L2 | partial | Captured simulate_call bytes are never decoded into semantics inputs. |
| S4.4 | records, diagnostics, forecast, watchdog, ops (2-GPU plan), checks/audit retention | L1/L2 | partial | The watchdog has no entry point; capture outputs are not consumed. |
| S4.6 | teacher_rl/evaluate.py, gate/superiority.py, gate/s2_gate.py | L1 | partial | Guards are neither registered nor bound; no glue runs decide; this path depends on the missing actual extraction. |
| Runbook-1 | s2_execution.validate_action_record, s2_readiness, s2_preflight | L1 | covered |  |
| Runbook-2 | s2_execution Coordinator, scripts/s2_lifecycle.py step, arm_watchdog argv | L1/L2 (L3) | covered | Has the authority-gate weakness (see safety). |
| Runbook-3 | scripts/vast_stop_watchdog.py (window ≤12 h, absolute deadline, no renewal) | L1/L2 | covered | Vast only. |
| Runbook-4 | bootstrap phase A, host probe, s2_execution._resource_mismatch | L2/L3 | partial | Guest VRAM, RAM and driver are not compared to the action record; XID delivery is unverified. |
| Runbook-5 | bootstrap phase E (per-file receipts, first delivery) | L2/L3 | owner_blocked |  |
| Runbook-6 | s2_actual, s2_capture, s2_launch.check_generated_configs, s2_gpu_sampler | L1/L2 | partial | No actual extraction, and diagnostic units cannot be declared. |
| Runbook-7 | s2_executor, scripts/s2_run_units.py, s2_ledger v2 | L1 (L3) | partial | Static budget file; no preflight gate. The full-sweep executor work in progress is excluded. |
| Runbook-8 | s2_guest_manifest, s2_pilot.verify_export, s2_bundle | L1 | partial | Transfer and incremental export are not orchestrated; the lifecycle trusts a typed export hash. |
| Runbook-9 | s2_execution stop/destroy/absence, charges; s2_cost reconcile | L1/L2 | covered |  |
| Runbook-10 | s2_execution cutoff_reached, retention authority, evidence_loss_risk | L1 | covered |  |

## Gaps

| Severity | Area | Summary | Evidence |
|---|---|---|---|
| high | S2.3/S2.5/S4.6 actual extraction | No path turns real AlpaSim output (metrics.parquet, ASL, results-summary.json) into W3 observations, describe_validity input, normalize_summary sidecars or evaluate_registered records. s2_extract admits only synthetic_source_contract_v1 and requires every sample to have valid True, but the pinned writer keeps invalid samples. s2_actual is descriptive only. | gate/s2_extract.py:27, :227; gate/s2_actual.py:44-50 (score_admitted false; no deviation computed) |
| high | S4.2-S4.4 capture to checks | Captured trainer evidence has no consumer outside teacher_rl/capture. records.py reads only payload, attempt and generation events. parameter_snapshot uses torch dtype strings and hash_scope local_shard, which is not rule 11 or the s4_param_snapshot schema that adapters.py:106 expects. Nothing decodes the SimulationReturn bytes in simulate_call. As a result E01-E04, E07, E10-E14 and E20 only run on hand-built inputs. | teacher_rl/capture/entrypoint.py:606-660, :705, :856-867, :946-960, :1009-1028; scripts/rehearse_s4.py:425-458 |
| high | CPU-09 guest runtime | No stage provisions or verifies the PhysLoop runner on the guest. The documented command is `python3 scripts/s2_guest_bootstrap.py` / `s2_run_units.py`, but gate/s2_bootstrap.py:64 imports tomllib and gate/s2_execution.py:118 imports datetime.UTC, both Python ≥3.11. The D-020 host is Ubuntu 22.04, whose system python3 is 3.10. s2_run_units also needs PyYAML and protobuf 7.36.2. uv is only version-checked (A6). No stage transfers the source snapshot, and scripts/s2_prepare_bundle.py is absent. | Local /usr/bin/python3 3.9: importing gate.s2_bootstrap raises ModuleNotFoundError tomllib; gate/s2_bootstrap.py:320-400 |
| medium | Authority gates (cross-module) | Only gate/s2_readiness.py:67 CLOSED_AUTHORITIES knows D-019 is closed, and no executing path consults it. s2_lifecycle needs only some `- **D-NNN ·` heading; s2_run_units checks free text bound to the packet; host_launch accepts any regex match. | scripts/s2_lifecycle.py:63-69, :197-215; scripts/s2_run_units.py:182-220; teacher_rl/capture/host_launch.py:241-247 |
| medium | S4.6 superiority binding | _bound_decisions checks delta_min, strata, metric, method and level, but never the decision's guards, guard_strata, gating_strata or look. improvement_decision_rule registers only rule_id and registration_ref. Tests aggregate decide(guards={}) to SUPPORTED. No code runs superiority.decide on evaluate_registered output, which reports superiority not_evaluated. | teacher_rl/evaluate.py:450-485; teacher_rl/registration.py:289; tests/test_s4_evaluate.py:313-326; gate/s2_gate.py:549 |
| medium | CPU-08 to runner | There are two next-unit checkers. The runner re-reads an operator-typed gate.s2_pilot request for every unit. Its used_usd and liability_usd are never derived from reconcile_cost, which is exposed only through s2_lifecycle next-unit. Headroom therefore does not shrink across units within a run. | scripts/s2_run_units.py:372-379, :506 |
| medium | CPU-13 report | build_s2_report copies caller-declared status, blockers and sha256 into each S2 item and never checks them against bundle_verification.manifest. I confirmed with a scratch script that a sha256 absent from a verified sealed bundle still gives S2.7 passed. | gate/s2_report.py:274-305; scratch review/completeness/report_binding.py |
| medium | CPU-11 to CPU-10 | Stop and destroy proceed on a typed export_verified observation that carries only a 64-hex hash. No verify_bundle or verify_export report is required before the instance storage is deleted. | gate/s2_execution.py:802-807, :992-1000 |
| medium | CPU-07/S2.7 producers | Only the nvidia-smi sampler exists. There is no DCGM .prom parser, and nothing produces the allocation segments partition or the useful-flagged attempt rows that summarize_telemetry needs. | grep finds no prom parser; gate/s2_telemetry.py:122-133 |
| medium | S2.6 superiority/sequential | run_sensitivity reports non-inferiority only. Superiority and falsifier power live in gate/superiority_power.py, which handles regular binary layouts only and is absent from the rehearsal and report. The registered glue and the S4 aggregation are fixed_sample only. | gate/s2_gate.py:85; teacher_rl/evaluate.py:109; gate/s2_report.py:88-92 |
| medium | S2.4 producers | There are no host-side runners for the fixed-input policy or fixed-pose renderer probes, and no procedure that produces the re-export snapshot for policy B. Only comparators exist, and launch only warns when the A and B snapshot paths are identical. | gate/s2_probes.py; gate/tensor_manifest.py; gate/s2_launch.py:566-569 |
| medium | E04 | Old log-probs (minibatch_started) and new ones (replay_forward) are captured, but no check pairs them under equal version, before the first optimizer step, at μ=1. | teacher_rl/capture/entrypoint.py:615-650; teacher_rl/parity.py |
| medium | E18 S4 watchdog | TrainingWatchdog has no CLI or host entry point and no producer of the watch spec (role PIDs/PGIDs). Only rehearse_s4 constructs it, with fakes. | teacher_rl/__main__.py:230-237; scripts/rehearse_s4.py:504 |
| medium | Section 11 final verification | rehearse_s2 never runs the suite, Ruff or whitespace checks inside the sealed snapshot. The build-spec §5 final clean frozen rebuild has not been produced: the 11-08 rebuild hashed uv.lock 725f2d1a… and pyproject 391ce016…, but the current files are 0a48c41f… and b8cc9949… (pyyaml and the s4 group were added since). | evidence/s2-clean-rebuild-20261108-v1/verification.json; shasum uv.lock pyproject.toml |
| low | CPU-00 | tests/test_s2_handoff.py is absent, and missing producer identities are not rejected. The rehearsal source manifest leaves out teacher_rl/ and git dirty state. | scripts/rehearse_s2.py:90 |
| low | Runbook step 6 | A diagnostic qualification unit cannot be declared or budgeted separately from the ten pilot jobs. R2 recommends at least one, for the camera check. | gate/s2_packet.py:621 |
| low | Runbook step 4 | Only provider-row fields are compared to the action record (gpu_name, num_gpus, disk, image, dph). The delivered guest VRAM, RAM, driver and CPU are not. | gate/s2_execution.py:848-865 |
| low | S4.1 registration | There is no slot for the response to AlpaGym issue #10 (trainable weights and Adam state may be bf16). The compile output only records an inherited master_dtype of float32. | teacher_rl/registration.py:86-93, :809-810 |
| low | S4 forecast | Every line must have gpu_count equal to 2, so a locked-confirmation evaluation on a separate 1-GPU S2 host with its own quote cannot be priced. | teacher_rl/forecast.py:178-207 |

## Safety

- High (creates, stops, destroys): `scripts/s2_lifecycle.py:63-69,:197-215`. The real Vast `step --execute` path only checks that a `- **D-NNN ·` heading exists in DECISIONS.md. With an admitted actual-scope record, closed D-019, D-020 and D-024 ("No rental … follows") all pass, which I confirmed with a scratch script. `validate_action_record` never checks that the decision authorizes this rental.
- Medium: `teacher_rl/capture/host_launch.py:241-247,:263-288`. The S4 launcher accepts any free-text `--action-authority-ref` that matches a regex. It then runs the upstream host CLI, which does the HF/NuRec preflight with the inherited HF_TOKEN, lets the Wizard download gated NuRec scenes, and starts GPU workers. It does not check `launch_hash`, the registration, `scope=actual` or the NuRec terms/acquire refs that `build_launch` requires.
- Medium: `scripts/s2_run_units.py:182-220`. Real units run on a free-text `action_record_ref` bound to the packet hash. Nothing checks DECISIONS.md or closed authority, and nothing requires a passed readiness report or preflight; the pilot preflight currently has 24 blockers.
- Medium (destroys unbound resources): `teacher_rl/watchdog.py:97,:369-384` runs `docker rm -f` on every container matching `name=wizard`, which is a substring filter, not the containers of this run.
- Medium: `gate/s2_execution.py:802-807,:992-1000` destroys the instance after a typed `export_verified` hash, without a verified export report. This risks losing the only copy of the evidence.
- Low: bootstrap phase E downloads and `docker login` require `--execute`, actual scope, non-SYNTHETIC `terms:`/`acquire:` refs and exact recompilation (`scripts/s2_guest_bootstrap.py:360-375`). The refs themselves are free text and never checked against DECISIONS.md.
- Tests: none runs a real vastai, docker, hf, huggingface-cli, nvidia-smi, nvidia-ctk or uv, or makes a network call. Providers and runners are faked or monkeypatched (`tests/test_s2_lifecycle_cli.py:283-289`, `tests/test_vast_stop_watchdog.py:382-393`). The probe tests use real bash/coreutils under a PATH sandbox, and `tests/test_vast_nvml_xid_probe.py:13-66` uses the real system c++. One fragility: `scripts/vast_stop_watchdog.py:380-418` `main` has no runner injection, so a future test that reached `watch()` would call the possibly authenticated vastai CLI.
- Credentials: no values reach artifacts, argv or logs. The Vast environment allowlists exclude VAST_API_KEY; HF/NGC secrets are read from private files and redacted; the capture records credential names only (`entrypoint.py:1122-1140`); bundles are scanned for canaries.
- Owner defaults: only `gate.s2 matrix --replicates` defaults to 3 (`gate/s2.py:290`, `s2_schedule.active_matrix_report`), and it only prints counts. Every other owner value is required and blocks when absent.

## Stale documentation

| File | Location | Correction |
|---|---|---|
| docs/S2_MINIMAL_PILOT_RUNBOOK.md | :77 (§4 step 3) | The watchdog window is now --max-window-seconds, equal to the action-record window, with a 12 h hard bound (scripts/vast_stop_watchdog.py:55), not 60 minutes. |
| docs/S2_CPU_TO_GPU_READINESS_SPEC.md | :254 (CPU-10 Constraints) | Same correction: the watchdog no longer rejects deadlines more than 60 minutes ahead; the bound is ≤12 h and equal to the window. |
| docs/S2_MINIMAL_PILOT_RUNBOOK.md | :71 (§3) | The 6-camera render request versus 4 consumed cameras now blocks before GPU work (s2_launch.check_generated_configs, s2_run_units step 4) until the owner's camera fix. |
| docs/S2_MINIMAL_PILOT_RUNBOOK.md | :105-109 (§6) | The uv 0.8.17 frozen rebuild passed in evidence/s2-clean-rebuild-20261108-v1, although that run predates the current lock. |
| docs/S2_INGESTION.md | :13-17 | The uv crash note and the 'clean sync unverified' statement are superseded by the 11-08 clean rebuild evidence. |
| docs/S2_INGESTION.md | :94-97 (Attempt ledger table) | Add the v2 events (schema_version 2, evidence_refs, completed reason process_completed) that gate/s2_executor emits (gate/s2_ledger.py:12-29). |
| docs/S2_CPU_HANDOFF_SPEC.md | :136-138 | The default ASL bound is now 2 GiB and read by streaming (gate/s2_asl.py:48, gate/s2_capture.py:60), not 256 MiB in RAM. |
| docs/S2_CPU_HANDOFF_SPEC.md | :209-211 | Same correction: the wrapper no longer limits ASL to 256 MiB in memory. |
| docs/S2_CPU_TO_GPU_READINESS_SPEC.md | :294 (CPU-13) | scripts/rehearse_s2.py --fixture complete is implemented and passes; it is no longer 'unavailable until implemented'. |
| docs/S2_CPU_TO_GPU_READINESS_SPEC.md | :236 (CPU-09 host-script repair) | The monitor and exporter scripts now take a digest reference and refuse DCGM conflicts (vast_monitor_control.sh:4-11; vast_exporter_control.sh:35-40). |
| docs/S2_CPU_TO_GPU_READINESS_SPEC.md | :116, :146, :160, :174, :190, :204 (interfaces) | The implemented signatures differ: validate_runtime_inventory(inventory); audit_population(manifest, roles, *, overlap_rule); describe_validity(ledger_report, observations, records); describe_repeats(observations, method, *, scope); run_sensitivity(design, scenarios, n_sim, seed); summarize_telemetry(samples, allocation, rollouts, *, clock_metadata, …). |
| docs/S2_CPU_TO_GPU_READINESS_SPEC.md | :218 (CPU-08) | Implemented as can_start_next_unit(ledger, limits, next_unit, *, now_epoch). The unit runner uses gate.s2_pilot.next_unit_check instead. |
| docs/S2_CPU_TO_GPU_READINESS_SPEC.md | :232 (CPU-09) | Implemented as build_execution_plan(inventory, action_record, *, phases) plus scripts/s2_guest_bootstrap.py. scripts/s2_prepare_bundle.py was never built. |
| docs/research/2026-11-08-codebase-build/build-spec.md | :10 (§0) | The verified rebuild covered uv.lock 725f2d1a… and pyproject 391ce016…. The current 0a48c41f… and b8cc9949… (pyyaml 6.0.3 and the s4 group) have not been rebuilt. |
| experiments/S2_PILOT_DRAFT.json | :220 (provider Nebius, wall_hours_cap 8) | Only a Vast lifecycle and cutoff exist; s2_execution.py:296-299 refuses any other provider. |
| docs/S2_MINIMAL_PILOT_RUNBOOK.md | :12-16 (§1) | Name scripts/rehearse_s2.py as the full CPU-13 rehearsal alongside the reduced rehearse_s2_pilot.py. |

## Owner inputs still open

- [pilot] Camera fix (R2 L1). Options: renderer_camera_override null or '4cam_1080', add cameras to the driver's use_cameras, or a driver patch. Every unit is blocked until this is chosen.
- [pilot] Harmonizer source: pinned_mount (revision and sha256) or upstream_default_download, plus the Harmonizer terms reference.
- [pilot] Licence and access actions with terms:/acquire: refs: Alpamayo-1.5-10B (OpenMDW-1.1), gated nvidia/Cosmos-Reason2-8B and its revision, NuRec dataset, NRE image (NGC), and private-host eligibility.
- [pilot] Five scene IDs with road and stratum mapping, target_population and road_independence_justification.
- [pilot] Requested horizon (n_sim_steps) and clipped-horizon/terminal semantics.
- [pilot] Registered execution order of the ten jobs.
- [pilot] Frozen registration: prediction (experiment file), decision rule, evaluated-path estimand (P-032 unadopted), seed-mode confirmation.
- [pilot] Retry-selection and missing-data rules. Today any retry blocks admission.
- [pilot] Preflight attestations: challenge_closed_verified, license_acceptance_by_owner, P-005 phase limits, quote date and exact GPU edition.
- [pilot] Provider choice: the draft names Nebius, but only a Vast lifecycle exists.
- [pilot] New rental action record and decision ID (D-019 is closed): offer, SKU, region, GPU edition and count, disk ≥300 GB, image, maximum hourly rate, window ≤12 h, lifecycle limits, retention mode and authority, export destination, cap headroom.
- [pilot] Unit-run authority bound to the packet; per-service Compose shm_size (NRE documents 64g).
- [pilot] XID/monitoring acceptance amendment if fault delivery stays unverified.
- [full] Release of the remaining S2 cap from the measured pilot forecast.
- [full] 441-scene list or amendment; roads and strata; target population; contamination/overlap rule; scene roles; S2 versus confirmation information boundary.
- [full] Primary endpoint, score definition, deviation/≥4 m convention, margins, alpha, fixed versus sequential looks.
- [full] Reliability method, estimand, hierarchy, interval, floor and degenerate convention; noise-floor prediction.
- [full] Validity definition for the A-valid/B-valid table.
- [full] Replicates (≥3), seed mode, the single Harmonizer setting, order of the 2,646 jobs.
- [full] Throughput stabilization rule; noise and cost predictions.
- [full] D-023 superiority design: falsifier per stratum or overall, named worse outcome, summed-level joint statement, fixed or sequential, δ_min, guard endpoints, strata and gating strata.
- [full] Re-export (policy B) control procedure, and any deterministic-control amendment.
- [S4-eng] S2 acceptance or an explicit stop/INSUFFICIENT; a separately capped S4 engineering admission.
- [S4-eng] Two-GPU rental action record: window, cap tier, failure reserve, export destination.
- [S4-eng] Cosmos-Reason2-8B licence and pinned revision; teacher OpenMDW-1.1; NuRec terms and acquire refs for the Wizard's scene download; NGC.
- [S4-eng] Learning-run settings: group_size, rollouts_per_trainer_step, learning_rate, lr_schedule, warmup steps and start factor, scene_set, optimizer_steps_per_run.
- [S4-eng] checkpoint_retention (save_freq, max_keep, export_safetensors); anchor or none.
- [S4-eng] scene_roles and overlap_rule, evaluated_policy_path, improvement_decision_rule, smallest_improvement_of_interest, independent_training_runs, run_aggregation (rule, k, family_alpha).
- [S4-eng] seed_modes, version_policy (on_policy, allowed_outdated_steps, prefetch), video, decode device (issue #9 risk).
- [S4-eng] e07_pass_criterion and parity_tolerances (log_density, action_trajectory).
- [S4-eng] Worker credential mode (strip_offline or inherit_disclosed); response to AlpaGym issue #10 (no slot exists).
- [S4-eng] prediction_ref experiment file; before scientific training, the definition of the KC-4 spread statistic.
