"""Preregistered CPU qualification with append-only, source-addressed evidence."""

import csv
import hashlib
import json
import platform
import subprocess
import time
import zipfile
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import beta

from gate.power import power_tables
from gate.sequential import SequentialGate
from gate.stats import continuous_mde
from gate.synthetic import generate, split_road_batches
from gate.verdict import verdict

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = "experiments/2026-10-31-s1-autonomous.md"


def configurations():
    common = {
        "strata": {"urban": 220, "highway": 221},
        "p_a": 0.07,
        "q_plus": 0.04,
        "q_minus": 0.02,
    }
    result = []
    for name, qp, qm, margin, truth in (
        ("equivalent", 0.02, 0.02, 0.02, "equal"),
        ("identical", 0.0, 0.0, 0.05, "equal"),
        ("boundary", 0.04, 0.02, 0.02, "bad"),
        ("harmful", 0.06, 0.02, 0.02, "bad"),
        ("gross", 0.27, 0.02, 0.02, "bad"),
    ):
        result.append(
            {
                "name": f"iid_{name}",
                "generator": {**common, "q_plus": qp, "q_minus": qm},
                "margin": margin,
                "truth": truth,
            }
        )
    result.append(
        {
            "name": "iid_one_bad",
            "generator": {**common, "effects": {"highway": (0.0, 0.0)}},
            "margin": 0.02,
            "truth": "bad",
        }
    )
    cluster = {
        **common,
        "strata": {"urban": 73, "highway": 74},
        "scenes_per_road": 3,
        "replicates": 3,
        "road_correlation": 0.8,
        "replicate_correlation": 0.6,
    }
    for name, qp in (("boundary", 0.04), ("harmful", 0.06)):
        result.append(
            {
                "name": f"cluster_{name}",
                "generator": {**cluster, "q_plus": qp},
                "margin": 0.02,
                "truth": "bad",
            }
        )
    result.append(
        {
            "name": "cluster_one_bad",
            "generator": {**cluster, "effects": {"highway": (0.0, 0.0)}},
            "margin": 0.02,
            "truth": "bad",
        }
    )
    result.append(
        {
            "name": "rare_harm",
            "generator": {
                **common,
                "strata": {"common": 432, "rare": 9},
                "q_plus": 0.0,
                "q_minus": 0.0,
                "effects": {"rare": (0.021, 0.0)},
            },
            "margin": 0.02,
            "truth": "bad",
        }
    )
    return result


def cp_upper(errors, trials, tail):
    return 1.0 if errors == trials else float(beta.ppf(1 - tail, errors + 1, trials - errors))


def _json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _sources():
    """Capture test inputs alongside code so the collected suite can replay.

    This changes future bundle contents only; existing evidence is immutable.
    Fixtures include provenance and license files, not only Python sources.
    """
    paths = [ROOT / EXPERIMENT, ROOT / "pyproject.toml", ROOT / "uv.lock"]
    for directory, pattern in (
        ("gate", "*.py"),
        ("tests", "*.py"),
        ("docs", "S1*.md"),
        ("experiments", "*.json"),
        ("experiments", "*.md"),
        ("tests/fixtures", "*"),
    ):
        paths.extend((ROOT / directory).rglob(pattern))
    return sorted(
        path
        for path in set(paths)
        if path.is_file()
        and not path.is_symlink()
        and "__pycache__" not in path.relative_to(ROOT).parts
        and path.suffix not in {".pyc", ".pyo"}
    )


def _write_tables(output, tables):
    _json(output / "power_tables.json", tables)
    directory = output / "tables"
    directory.mkdir()
    for name, rows in tables.items():
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (directory / f"{name}.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        k: json.dumps(v) if isinstance(v, (dict, list, tuple)) else v
                        for k, v in row.items()
                    }
                )


def _plot(output, summaries):
    # Standalone SVG, no plotting dependency or synthetic screenshot.
    width, height = 1050, 100 + 34 * len(summaries)
    colors = {"GO": "#26734d", "NO-GO": "#b23b35", "INSUFFICIENT": "#c69a32"}
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" font-size="13" fill="#202020">',
        '<text x="20" y="25" font-size="19">Synthetic gate outcomes by registered cell</text>',
        '<text x="20" y="48">Green GO · red NO-GO · amber INSUFFICIENT; no real-policy claims</text>',
    ]
    for i, (name, row) in enumerate(sorted(summaries.items())):
        y = 75 + i * 34
        parts.append(f'<text x="20" y="{y + 14}">{name}</text>')
        x = 300.0
        for state, color in colors.items():
            count = row["counts"].get(state, 0)
            w = 600 * count / row["trials"]
            parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="20" fill="{color}"/>')
            x += w
        parts.append(f'<text x="915" y="{y + 14}">n={row["trials"]}</text>')
    parts.append("</g></svg>")
    (output / "outcomes.svg").write_text("\n".join(parts))


def run_experiment(output, trials=1000, master_seed=20261031):
    """Never overwrite evidence; interrupted runs remain distinguishable as incomplete."""
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be positive")
    if isinstance(master_seed, bool) or not isinstance(master_seed, int) or master_seed < 0:
        raise ValueError("seed must be nonnegative")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("evidence directory exists; choose a new run ID")
    cases = configurations()
    config = {
        "schema_version": 1,
        "trials": trials,
        "master_seed": master_seed,
        "alpha": 0.05,
        "sequential_weights": [0.2, 0.3, 0.5],
        "cases": cases,
        "qualification": trials == 1000 and master_seed == 20261031,
        "estimand": "equal road / equal scene / equal replicate",
        "seed_mode": "synthetic_seeded",
        "wall_clock_cap_seconds": 1800,
    }
    source_bytes = {str(path.relative_to(ROOT)): path.read_bytes() for path in _sources()}
    source_hashes = {
        name: hashlib.sha256(contents).hexdigest() for name, contents in source_bytes.items()
    }
    output.mkdir(parents=True)
    _json(output / "config.json", config)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "git_hash": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "source_sha256": source_hashes,
        "config_sha256": hashlib.sha256((output / "config.json").read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "device": "local CPU",
        "gpu_hours": 0,
        "cloud_cost_usd": 0,
        "replicates": "per config",
        "force_determinism": "not applicable: synthetic NumPy RNG",
        "alpasim_branch": "not applicable",
        "alpasim_sha": "not applicable",
        "evaluator_sha": "source hashes",
        "n_sim_steps_requested": "not applicable",
        "n_sim_steps_actual": "not applicable",
        "simulator_configuration": "not applicable",
        "driver_settings": "not applicable",
        "gpu_edition": "not applicable",
        "scene_list": "deterministically regenerated from trial seed and config",
        "score_definition": "paired binary failure, B minus A",
        "deviation_convention": "include and report separately; no deviations in generator",
        "license_inputs": "synthetic only",
    }
    _json(output / "manifest.json", manifest)
    with zipfile.ZipFile(output / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, contents in source_bytes.items():
            archive.writestr(name, contents)
    started = time.monotonic()
    totals = {}
    with (output / "trials.jsonl").open("x") as stream:
        for case_index, case in enumerate(cases):
            for trial in range(trials):
                if time.monotonic() - started > 1800:
                    raise TimeoutError("preregistered CPU wall-clock cap reached")
                streams = np.random.SeedSequence([master_seed, case_index, trial]).spawn(2)
                seed, order_seed = (int(s.generate_state(1, dtype=np.uint64)[0]) for s in streams)
                rows = generate(seed, **case["generator"])
                required = tuple(sorted(case["generator"]["strata"]))
                fixed = verdict(rows, case["margin"], required_strata=required)
                sequential = SequentialGate(required, case["margin"])
                looks = []
                for batch in split_road_batches(rows, K=3, seed=order_seed):
                    active = set(sequential.active_strata)
                    seq_result = sequential.update([r for r in batch if r["stratum"] in active])
                    looks.append({"look": sequential.look, "result": asdict(seq_result)})
                    if sequential.terminal:
                        break
                results = {"fixed": fixed, "sequential": sequential.result}
                if case["name"] == "rare_harm":
                    results["pooled_fixed"] = verdict(
                        [{**r, "stratum": "pooled"} for r in rows],
                        case["margin"],
                        required_strata=("pooled",),
                    )
                for mode, result in results.items():
                    key = case["name"] + "/" + mode
                    truth = "different_pooled_estimand" if mode == "pooled_fixed" else case["truth"]
                    cell = totals.setdefault(
                        key,
                        {
                            "counts": Counter(),
                            "trials": 0,
                            "truth": truth,
                            "any_stratum_insufficient": 0,
                        },
                    )
                    cell["counts"][result.overall] += 1
                    cell["trials"] += 1
                    cell["any_stratum_insufficient"] += int(
                        "INSUFFICIENT" in result.strata.values()
                    )
                record = {
                    "case": case["name"],
                    "trial": trial,
                    "seed": seed,
                    "order_seed": order_seed,
                    "results": {m: asdict(r) for m, r in results.items()},
                    "looks": looks,
                }
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            print(f"completed {case['name']}: {trials} trials", flush=True)
    # One simultaneous family for all harmful and equal-policy calibration cells.
    family_size = sum(cell["truth"] in ("bad", "equal") for cell in totals.values())
    for cell in totals.values():
        cell["counts"] = dict(cell["counts"])
        if cell["truth"] in ("bad", "equal"):
            error = "GO" if cell["truth"] == "bad" else "NO-GO"
            count = cell["counts"].get(error, 0)
            cell.update(
                error_type="false_GO" if error == "GO" else "false_NO_GO",
                errors=count,
                one_sided_upper=cp_upper(count, trials, 0.05 / family_size),
                confidence_tail=0.05 / family_size,
            )
    calibration = all(cell.get("one_sided_upper", 0) <= 0.05 for cell in totals.values())
    controls = {
        "identical_all_go": all(
            totals[f"iid_identical/{m}"]["counts"].get("GO", 0) == trials
            for m in ("fixed", "sequential")
        ),
        "gross_detected": all(
            totals[f"iid_gross/{m}"]["counts"].get("NO-GO", 0) / trials >= 0.95
            for m in ("fixed", "sequential")
        ),
        "rare_no_go": all(
            totals[f"rare_harm/{m}"]["counts"].get("GO", 0) == 0 for m in ("fixed", "sequential")
        ),
        "pooling_exposes_hidden_loss": totals["rare_harm/pooled_fixed"]["counts"].get("GO", 0)
        / trials
        >= 0.80,
    }
    predictions = {
        "PA1_rare_insufficient": totals["rare_harm/fixed"]["counts"].get("INSUFFICIENT", 0) / trials
        >= 0.99,
        "PA2_equivalent_insufficient": totals["iid_equivalent/fixed"]["counts"].get(
            "INSUFFICIENT", 0
        )
        / trials
        >= 0.8,
        "PA3_pooling": controls["pooling_exposes_hidden_loss"],
        "PA4_planning_baselines": {
            "single_replicate": continuous_mde(0.2, 441) / 0.2,
            "m3_rho06": continuous_mde(0.2 * (2.2 / 3) ** 0.5, 441) / 0.2,
        },
        "PA5_stratified_insufficient": totals["iid_equivalent/fixed"]["any_stratum_insufficient"]
        / trials
        >= 0.8,
        "PA6_controls": controls["identical_all_go"] and controls["gross_detected"],
        "PA7_observed_false_go": all(
            c.get("errors", 0) / trials <= 0.01 for c in totals.values() if c["truth"] == "bad"
        ),
    }
    tables = power_tables()
    _write_tables(output, tables)
    _plot(output, totals)
    summary = {
        "status": "complete",
        "qualification": config["qualification"],
        "cells": totals,
        "simultaneous_family_size": family_size,
        "calibration_pass": calibration,
        "controls": controls,
        "predictions": predictions,
        "experiment_pass": calibration and all(controls.values()),
        "elapsed_seconds": time.monotonic() - started,
        "note": "Not an unconditional correctness proof or authorization to spend in S2.",
    }
    _json(output / "summary.json", summary)
    lines = [
        "# S1 synthetic qualification results",
        "",
        (
            f"Experiment pass: **{summary['experiment_pass']}**. Registered qualification: "
            f"**{config['qualification']}**. CPU only; cloud cost $0."
        ),
        "",
        (
            "The operational method is the conservative road-level KL gate, not the Gaussian "
            "planning design. Each endpoint is separately one-sided. Raw trials and source hashes "
            "are adjacent to this report."
        ),
        "",
        "| Cell | GO | NO-GO | INSUFFICIENT | Error upper bound |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, cell in sorted(totals.items()):
        counts = cell["counts"]
        upper = (
            f"{cell['one_sided_upper']:.6f}"
            if "one_sided_upper" in cell
            else "not the same estimand"
        )
        lines.append(
            f"| {name} | {counts.get('GO', 0)} | {counts.get('NO-GO', 0)} | "
            f"{counts.get('INSUFFICIENT', 0)} | {upper} |"
        )
    lines += [
        "",
        (
            "The error bounds use exact binomial inversion with a simultaneous 95% family "
            f"over {family_size} registered calibration cells. Monte Carlo supports only these "
            "configurations; the mathematical guarantee relies on independent road clusters and "
            "a fixed sampling design."
        ),
        "",
        (
            "Pooled rare-harm GO is not a false-GO for the pooled estimand. It demonstrates why "
            "a pooled claim cannot replace the registered per-stratum claim."
        ),
        "",
        (
            "Gaussian power tables, IID Tango and independent-Poisson rate-ratio calculations "
            "are planning/reference outputs. S2 must measure actual dependence and score variability."
        ),
        "",
        (
            "VI: Các số trên thuộc dữ liệu giả với sự thật đã biết. Không phải kết quả của "
            "Alpamayo hoặc bằng chứng lái xe ngoài đời."
        ),
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    hashes = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    _json(output / "checksums.json", hashes)
    return summary
