# contribution-review-20261105-v1

Evidence for docs/research/2026-11-05-contribution-review/ (P-032, proposal only).
No GPU run, rental, license acceptance or gated download produced any file here.

- pytest-20261105.log: `uv run --frozen pytest -q` at HEAD 3741d92 (756 passed).
- root-calc/: coordinating-agent scripts and outputs. crn_power.py and bound_compare.py import the
  locked gate.verdict read-only; run from the repo root with `uv run --frozen python <script>`.
  oracles.py and seed_overlap.py are pure arithmetic (`python3 -I`).
- lane-A-derive/: lane A parameter/FLOP arithmetic; run
  `python3 -I lane-A-derive/derive.py hf-public-configs/a15_config.json hf-public-configs/qwen3vl8b_config.json`.
- lane-C-scripts/, lane-C-power_out/, lane-C-upstream_csv/: lane C power, missing-data and catalog
  scripts with outputs and public AlpaSim catalog CSV copies (e2e_challenge 297b4d0). The scripts were
  written against scratchpad paths; pass these directories as arguments.
- hf-public-configs/: public Hugging Face config JSON (no weights).
- vast-public-offers/: public unauthenticated Vast offer search, 2026-11-05 16:34 UTC, no API key.
All inputs are illustrative where marked; nothing here is a simulator measurement.
