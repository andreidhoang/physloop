# S2 report (synthetic)

Engineering status, milestone labels, evidence integrity and human authorization are separate; none of them is an admission.

## Milestone labels

| Label | Status | Reason |
|---|---|---|
| CPU_REHEARSED | passed |  |
| PACKET_COMPLETE | not_applicable | synthetic or fixture readiness cannot complete a real packet |
| HOST_QUALIFIED | qualify_on_host | Only a named delivered host and actual artifacts establish this label |
| ACTUAL_INPUT_QUALIFIED | qualify_on_host | Only a named delivered host and actual artifacts establish this label |
| S2_ACCEPTED | not_run | Registered controls, reliability, validity, power and measured forecast on actual evidence, judged by Huy; unreachable from CPU inputs |
| S2_STOPPED_OR_INSUFFICIENT | not_run | A measured study result; no actual study has run |

## Evidence integrity

Bundle: passed (seal sealed); authenticity not verified.

## Human authorization

Owner decisions: passed; action scope: passed (readiness scope synthetic); not authenticated; no launch authority.

## S2 card items

| Item | Title | Engineering | Scientific | Missing evidence | Blockers |
|---|---|---|---|---|---|
| S2.1 | Rights, revisions and registration | unverified | blocked |  | dcgm_exporter_run_long_stage_absent, hf_offline_cache_resolution_unverified |
| S2.2 | Bounded cost/protocol pilot | blocked | blocked |  | job_not_completed, retry_selection_unregistered |
| S2.3 | Complete extraction and known answers | passed | blocked |  |  |
| S2.4 | Frozen-teacher reliability and controls | blocked | blocked |  | method_unregistered |
| S2.5 | Validity and contamination | unverified | blocked |  | overlap_rule_unregistered |
| S2.6 | Operational power and confirmation design | passed | blocked |  |  |
| S2.7 | Cost and necessary telemetry | passed | blocked |  |  |

## Unverified evidence

Rule: a row is bound only when a passed, sealed verify_bundle manifest lists its path with its sha256 and the row names its producer.

- S2.1: bootstrap_plan: declared_passed_with_blockers
- S2.5: population: declared_passed_with_blockers

## Huy-owned inputs (not filled by agents)

- S2.1: target population; road/scene/stratum rules; metrics; margins; validity and terminal semantics; seed mode; Harmonizer setting and source; control horizon/frequency; run method/topology; reliability floors; scientific predictions; licence/access actions
- S2.2: five-scene registration; execution order; pilot prediction; camera fix; rental action record
- S2.3: retry-selection rule; missing-data rule; terminal semantics
- S2.4: reliability method, estimand and floor; seed mode; replicate count; scene list and order
- S2.5: validity definition; overlap/exclusion rule; road and stratum assignment
- S2.6: margins; strata; superiority rule; evaluated-path estimand; information boundary
- S2.7: stabilization criterion; noise and cost predictions

## Next actions

- owner: none listed
- agent: none listed
- host: none listed

## Counts

- bundle_entries: 106
- full_sweep_units: 2646
- pilot_attempts: 11
- pilot_units: 10
- replayed_analyses: 12

## Not claimed

- no host, renderer, model or Harmonizer qualification
- no scientific admission, release verdict or S2 acceptance
- no rental, funding, licence or publication authority
- no per-mile rate, safety, human-parity or real-world validation claim
- synthetic numbers are fixtures, not measurements
