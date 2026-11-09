"""Evaluator-path known answers on pinned public AlpaSim aggregation source (CPU-12).

L2: the references are unmodified, hash-checked public files in
``tests/fixtures/alpasim_s2_raw/`` (NVlabs/alpasim ``e121e59141dc391f6c295513d6dadc840a419149``,
Apache-2.0, LICENSE alongside): ``processing_e121e591.py.txt``, ``modifiers_e121e591.py.txt``,
``main_e121e591.py.txt`` and ``scene_score_e121e591.py.txt`` (``src/eval/src/eval/aggregation/``).
Only named AST nodes are compiled, into restricted namespaces; the full evaluator, Polars,
datasets and GPU libraries are never imported. Success here is AST-extracted helper evidence;
it is never whole-evaluator or runtime qualification.

Coverage tables (``COVERAGE``, ``MODIFIERS_COVERAGE``, ``MAIN_COVERAGE`` and
``SCENE_SCORE_COVERAGE`` below are the machine-checked copies; every top-level definition of
each pinned file is listed):

| Pinned definition | Status | Dependency stubs | Untested integration |
|---|---|---|---|
| processing ``DEFAULT_MODIFIERS`` | executed (symbolic and with the pinned modifier classes) | ``pl`` stub (``Expr``/``MiniFrame``) | Polars itself |
| processing ``_add_progress_clipped_rel_metric`` | executed | ``MiniFrame`` + ``pl`` stub (col/alias/concat/DataFrame) | real Polars frames |
| processing ``_json_safe`` | executed | none (pure Python, restricted builtins) | none |
| processing ``aggregate_and_write_metrics_results_txt`` | executed (partial) | missing-offroad ``if`` node only, with ``MiniFrame``; modifier loop mirrored with ``.apply`` | pivot, ``__call__`` bookkeeping, time/clip/rollout aggregation not run |
| processing ``write_results_summary_json`` | structural | none | ``score_rollout`` not run |
| processing, every other definition | not_run | none | Polars-only frame plumbing, plotting and text output |
| modifiers ``remove_timesteps_before_or_after_event``, ``RemoveTimestepsBeforeEvent``, ``RemoveTimestepsAfterEvent`` | executed | ``pl`` stub (sort/with_columns/filter/drop; filter/first/over/is_null) | Polars itself |
| modifiers ``MetricAggregationModifiers``, ``AddCombinedEvent`` | executed (partial) | as above; ``apply`` only | ``__call__`` (anti-join row counts, logging), ``apply_agg_function_df`` |
| modifiers ``get_removed_rows``, ``RemoveTrajectoryWithEvent`` | not_run | none | not on the pinned default path |
| main ``_run_aggregation_core`` | executed (partial) | ``modifiers = [...]`` assignment only, ``cfg`` stub | config loading, job discovery, video handling |
| main, every other definition | not_run / structural | none | file system, subprocess (ffmpeg), CLI |
| scene_score, every definition | not_run | none | vendored and hash-checked for a later scorer test only |

The >= 4 m threshold is a config input, not a code default: ``eval/schema.py`` declares
``max_dist_to_gt_trajectory: float = MISSING`` and ``src/wizard/configs/base_config.yaml`` at
the pin sets ``4.0``. Neither file is vendored; the tests pass ``4.0`` explicitly.

Test-only Polars stub (unavoidable: Polars is not in the locked environment). ``Expr``,
``evaluate`` and ``MiniFrame`` model only the operations the executed nodes call, with Polars
null semantics made explicit: a comparison or arithmetic with ``None`` yields ``None``; ``|``
is Kleene (``True`` if either side is ``True``, else ``None`` if either is ``None``);
``DataFrame.filter`` keeps only rows whose predicate is ``True``; ``col.filter(p).first()
.over(k)`` is the first value per group in frame order where ``p`` is ``True``, else ``None``;
``sort`` is lexicographic on the named keys. ``test_whole_evaluator_path_remains_not_run``
fails if Polars arrives or a still-unvendored dependency is vendored, so the tables must be
revisited rather than silently relabelled.

Findings preserved as upstream behaviour (not patched): upstream zero-fills a missing
``offroad`` column before the modifiers and scorer run (PhysLoop must not: normalization
stays blocked without verified required metrics); the Polars path pivots ``values`` only and
never reads the ``valid`` mask, while ``MetricReturn.aggregate`` excludes invalid samples;
warm-up removal keeps every row from the first eval-relevant step on, and a trajectory with no
eval-relevant step keeps all its rows; the first-incident truncation keeps the incident row,
drops every later row, and fires on any collision (``collision_any``), not only an at-fault
one; the >= 4 m modifier runs after the default chain, keeps the first crossing row and drops
every later row, so a later collision or a larger deviation never reaches the summary
(PhysLoop reports >= 4 m rollouts separately from the preserved full stream).
"""

import ast
import builtins
import dataclasses
import hashlib
import importlib.util
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gate.alpasim_adapter import SUPPORTED_EVALUATOR_SHA, normalize_summary

ROOT = Path(__file__).parent / "fixtures" / "alpasim_s2_raw"
PROCESSING = "processing_e121e591.py.txt"
PROCESSING_SHA256 = "c6e45ff7802b1364333c0bc115c77077e3a61098cb21af0fdacbcc24486bc2cb"
PROCESSING_GIT_BLOB = "2abd4b093d1801c6fb9a66a868ec05c6a01518c0"
DATA = "data_e121e591.py.txt"
DATA_SHA256 = "aef422e48cb161b09df227c4ad338ff4bef152a3154358e8ae0051ba95a0a5b2"
COMMIT = "e121e59141dc391f6c295513d6dadc840a419149"

COVERAGE = {
    "logger": "not_applicable",
    "DEFAULT_MODIFIERS": "executed_symbolic",
    "_add_progress_clipped_rel_metric": "executed",
    "_combine_run_uuids_deterministically": "not_run",
    "add_rollout_and_trajectory_uids": "not_run",
    "aggregate_over_clips": "not_run",
    "write_metrics_results_txt": "not_run",
    "_json_safe": "executed",
    "write_results_summary_json": "structural",
    "rename_legend_handles": "not_run",
    "plot_metrics_results": "not_run",
    "UnprocessedMetricsDFs": "not_run",
    "ProcessedMetricDFs": "not_run",
    "get_avg_dist_between_incidents": "not_run",
    "aggregate_and_write_metrics_results_txt": "executed_partial",
}
STATUSES = {"executed", "executed_symbolic", "executed_partial", "structural", "not_run",
            "not_applicable"}
MODIFIERS = "modifiers_e121e591.py.txt"
MAIN = "main_e121e591.py.txt"
SCENE_SCORE = "scene_score_e121e591.py.txt"
# file -> (repository path, git blob SHA-1, SHA-256); receipts proven against the pinned clone.
AGGREGATION_SOURCES = {
    MODIFIERS: ("src/eval/src/eval/aggregation/modifiers.py",
                "025537b9244036203fe31ddfe036d965368061cb",
                "d4bab741579f3babdca3896b64001db3f7ee4934731495ddd9145619fd22b5bf"),
    MAIN: ("src/eval/src/eval/aggregation/main.py",
           "a4ab0bd736f194845fa6f25a0b1d10a276d75023",
           "a9d52262f5bce0d47ddc3e5b1f50290bf780d356cde41ca303acbaa38f717374"),
    SCENE_SCORE: ("src/eval/src/eval/aggregation/scene_score.py",
                  "4bdc177b5f26996f99eb4db1cc16123a713b3ed3",
                  "e5231c89dc3eb6eff5e1fb33f59978a940c0e16880fa019c9feb2be4b6792714"),
}
MODIFIERS_COVERAGE = {
    "logger": "not_applicable",
    "get_removed_rows": "not_run",
    "MetricAggregationModifiers": "executed_partial",
    "remove_timesteps_before_or_after_event": "executed",
    "RemoveTimestepsAfterEvent": "executed",
    "RemoveTimestepsBeforeEvent": "executed",
    "AddCombinedEvent": "executed_partial",
    "RemoveTrajectoryWithEvent": "not_run",
}
MAIN_COVERAGE = {
    "WIZARD_FILE": "not_applicable",
    "VIDEO_FILE_NAME_FORMAT": "not_applicable",
    "root_logger": "not_applicable",
    "logger": "not_applicable",
    "CONCAT_VIDEO_NAME": "not_applicable",
    "_collect_metrics_from_job_dir": "not_run",
    "_aggregate_metrics": "structural",
    "_run_aggregation_core": "executed_partial",
    "run_aggregation_from_runtime": "not_run",
    "_discover_job_dirs": "not_run",
    "_speed_up_video": "not_run",
    "_concatenate_videos": "not_run",
    "_aggregate_eval_videos": "not_run",
    "main": "not_run",
}
SCENE_SCORE_COVERAGE = dict.fromkeys(
    ("REQUIRED_SCENE_SCORE_METRICS", "SceneScoreResult", "_required_finite_float",
     "_required_scene_score_values", "_progress_score", "_hard_failure_reason", "score_rollout"),
    "not_run",
)
# Still absent from the fixture set; the sentinel fails if any of them is vendored.
UNVENDORED_DEPENDENCIES = (
    "src/eval/src/eval/schema.py",
    "src/eval/src/eval/aggregation/failed_rollouts.py",
    "src/eval/src/eval/aggregation/telemetry.py",
    "src/eval/src/eval/aggregation/utils.py",
)
MAX_DIST_TO_GT_TRAJECTORY_M = 4.0  # base_config.yaml at the pin; schema.py has no default.


# --- pinned bytes ----------------------------------------------------------------------


def _source(name, digest):
    raw = (ROOT / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest, "pinned upstream source bytes changed"
    return raw


def _tree():
    return ast.parse(_source(PROCESSING, PROCESSING_SHA256), filename=PROCESSING)


def _top_level(tree, name):
    found = [
        node
        for node in tree.body
        if getattr(node, "name", None) == name
        or (isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in node.targets))
    ]
    assert len(found) == 1, name
    return found[0]


def _function_body_statement(function, predicate):
    found = [statement for statement in function.body if predicate(statement)]
    assert len(found) == 1
    return found[0], function.body.index(found[0])


def _execute(nodes, namespace, label):
    # Execute only named nodes from content-addressed public source.
    code = compile(ast.Module(body=list(nodes), type_ignores=[]), label, "exec")
    exec(code, namespace)  # noqa: S102
    return namespace


def test_processing_source_matches_provenance_git_blob_and_license():
    raw = _source(PROCESSING, PROCESSING_SHA256)
    provenance = json.loads((ROOT / "provenance.json").read_text())
    assert provenance["commit"] == COMMIT
    [entry] = [e for e in provenance["sources"] if e["file"] == PROCESSING]
    assert entry["path"] == "src/eval/src/eval/aggregation/processing.py"
    assert entry["sha256"] == PROCESSING_SHA256
    assert entry["git_blob_sha"] == PROCESSING_GIT_BLOB
    blob = b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
    assert hashlib.sha1(blob).hexdigest() == PROCESSING_GIT_BLOB
    assert COMMIT in entry["source_url"]
    assert raw.startswith(b"# SPDX-License-Identifier: Apache-2.0\n")
    assert provenance["license"] == "Apache-2.0"
    license_raw = (ROOT / "LICENSE").read_bytes()
    assert hashlib.sha256(license_raw).hexdigest() == provenance["license_file"]["sha256"]


def test_s2_ground_truth_snapshot_is_the_one_with_existing_endpoint_goldens():
    # tests/test_alpasim_ground_truth_upstream.py holds the endpoint goldens for these bytes.
    s2 = (ROOT / "ground_truth_e121e591.py.txt").read_bytes()
    goldens = (Path(__file__).parent / "fixtures" / "alpasim" / "ground_truth_e121e591.py.txt")
    assert s2 == goldens.read_bytes()


def _top_level_names(tree):
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names.extend(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _aggregation_tree(name):
    _, _, digest = AGGREGATION_SOURCES[name]
    return ast.parse(_source(name, digest), filename=name)


def test_coverage_table_lists_every_top_level_definition():
    assert sorted(_top_level_names(_tree())) == sorted(COVERAGE)
    tables = {MODIFIERS: MODIFIERS_COVERAGE, MAIN: MAIN_COVERAGE,
              SCENE_SCORE: SCENE_SCORE_COVERAGE}
    for name, table in tables.items():
        assert sorted(_top_level_names(_aggregation_tree(name))) == sorted(table), name
    for table in (COVERAGE, *tables.values()):
        assert set(table.values()) <= STATUSES


@pytest.mark.parametrize("name", AGGREGATION_SOURCES)
def test_aggregation_sources_match_provenance_git_blob_and_license(name):
    path, git_blob, digest = AGGREGATION_SOURCES[name]
    raw = _source(name, digest)
    blob = b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
    assert hashlib.sha1(blob).hexdigest() == git_blob
    provenance = json.loads((ROOT / "provenance.json").read_text())
    [entry] = [e for e in provenance["sources"] if e["file"] == name]
    assert (entry["path"], entry["git_blob_sha"], entry["sha256"]) == (path, git_blob, digest)
    assert entry["source_url"] == f"https://github.com/NVlabs/alpasim/blob/{COMMIT}/{path}"
    assert raw.startswith(b"# SPDX-License-Identifier: Apache-2.0\n")
    assert b"NVIDIA Corporation" in raw.splitlines()[1]


def test_whole_evaluator_path_remains_not_run():
    """Runtime blocker sentinel: revisit the coverage tables if Polars or a dependency arrives."""
    assert importlib.util.find_spec("polars") is None, "Polars present: revisit COVERAGE"
    provenance = json.loads((ROOT / "provenance.json").read_text())
    vendored = {entry["path"] for entry in provenance["sources"]}
    assert {path for path, _, _ in AGGREGATION_SOURCES.values()} <= vendored
    assert not vendored & set(UNVENDORED_DEPENDENCIES), "dependency vendored: revisit COVERAGE"
    imports = {
        node.module: {alias.name for alias in node.names}
        for node in _tree().body
        if isinstance(node, ast.ImportFrom)
    }
    assert imports["eval.aggregation.modifiers"] >= {
        "RemoveTimestepsAfterEvent",
        "RemoveTimestepsBeforeEvent",
        "AddCombinedEvent",
    }
    assert imports["eval.aggregation.scene_score"] == {"score_rollout"}


# --- dependency stubs (test-only) -------------------------------------------------------


class Expr:
    """Symbolic stand-in for the subset of ``polars.Expr`` the pinned nodes build."""

    def __init__(self, op, *args):
        self.op, self.args = op, args

    def _binary(self, op, other):
        return Expr(op, self, other if isinstance(other, Expr) else Expr("lit", other))

    def __gt__(self, other):
        return self._binary("gt", other)

    def __ge__(self, other):
        return self._binary("ge", other)

    def __le__(self, other):
        return self._binary("le", other)

    def filter(self, predicate):
        return Expr("filter", self, predicate)

    def first(self):
        return Expr("first", self)

    def is_null(self):
        return Expr("is_null", self)

    def cast(self, dtype):
        return Expr("cast", self, dtype)

    def __or__(self, other):
        return self._binary("or", other)

    def __sub__(self, other):
        return self._binary("sub", other)

    def __truediv__(self, other):
        return self._binary("div", other)

    def min(self):
        return Expr("min", self)

    def over(self, key):
        return Expr("over", self, key)

    def alias(self, name):
        return Alias(self, name)

    def describe(self):
        if self.op == "col":
            return f"col({self.args[0]})"
        if self.op == "lit":
            return repr(self.args[0])
        if self.op == "over":
            return f"{self.args[0].describe()}.over({self.args[1]})"
        if self.op == "min":
            return f"{self.args[0].describe()}.min()"
        symbol = {"gt": ">", "ge": ">=", "le": "<=", "or": "|", "sub": "-", "div": "/"}[self.op]
        return f"({self.args[0].describe()} {symbol} {self.args[1].describe()})"


class Alias:
    def __init__(self, expr, name):
        self.expr, self.name = expr, name

    def cast(self, dtype):
        return Alias(self.expr.cast(dtype), self.name)


def _null_propagating(function):
    return lambda x, y: None if x is None or y is None else function(x, y)


def _kleene_or(x, y):
    if x is True or y is True:
        return True
    return None if x is None or y is None else False


def _first_true(values, mask, groups, group):
    rows = zip(values, mask, groups, strict=True)
    return next((v for v, m, g in rows if g == group and m is True), None)


def evaluate(expr, frame):
    n = len(frame)
    if expr.op == "col":
        return list(frame.data[expr.args[0]])
    if expr.op == "lit":
        return [expr.args[0]] * n
    if expr.op == "is_null":
        return [value is None for value in evaluate(expr.args[0], frame)]
    if expr.op == "cast":
        inner, dtype = expr.args
        assert dtype is PL.Float64
        return [None if value is None else float(value) for value in evaluate(inner, frame)]
    if expr.op == "over":
        inner, key = expr.args
        groups = frame.data[key]
        if inner.op == "first":
            # col.filter(predicate).first().over(key), in frame order per group.
            selection = inner.args[0]
            assert selection.op == "filter"
            values = evaluate(selection.args[0], frame)
            mask = evaluate(selection.args[1], frame)
            firsts = {g: _first_true(values, mask, groups, g) for g in set(groups)}
            return [firsts[g] for g in groups]
        assert inner.op == "min"
        values = evaluate(inner.args[0], frame)
        minima = {g: min(v for v, h in zip(values, groups, strict=True) if h == g)
                  for g in set(groups)}
        return [minima[g] for g in groups]
    left, right = (evaluate(arg, frame) for arg in expr.args)
    function = {
        "gt": _null_propagating(lambda x, y: x > y),
        "ge": _null_propagating(lambda x, y: x >= y),
        "le": _null_propagating(lambda x, y: x <= y),
        "or": _kleene_or,
        "sub": _null_propagating(lambda x, y: x - y),
        "div": _null_propagating(lambda x, y: x / y),
    }[expr.op]
    return [function(x, y) for x, y in zip(left, right, strict=True)]


class MiniFrame:
    """Column-dict stand-in for the few ``polars.DataFrame`` methods the nodes call."""

    def __init__(self, data):
        lengths = {len(v) for v in data.values()}
        assert len(lengths) <= 1
        self.data = {k: list(v) for k, v in data.items()}

    @property
    def columns(self):
        return list(self.data)

    def __len__(self):
        return len(next(iter(self.data.values()), []))

    def with_columns(self, *aliases):
        data = dict(self.data)
        for alias in aliases:
            data[alias.name] = evaluate(alias.expr, self)
        return MiniFrame(data)

    def _rows(self, indices):
        return MiniFrame({k: [v[i] for i in indices] for k, v in self.data.items()})

    def sort(self, by):
        return self._rows(sorted(range(len(self)),
                                 key=lambda i: tuple(self.data[c][i] for c in by)))

    def filter(self, predicate):
        keep = evaluate(predicate, self)
        return self._rows([i for i, kept in enumerate(keep) if kept is True])

    def drop(self, name):
        assert name in self.data
        return MiniFrame({k: v for k, v in self.data.items() if k != name})


class PL:
    Expr = Expr
    DataFrame = MiniFrame
    Float64 = object()  # dtype sentinel; only ``cast(pl.Float64)`` is modelled

    @staticmethod
    def col(name):
        return Expr("col", name)

    @staticmethod
    def lit(value):
        return Expr("lit", value)

    @staticmethod
    def concat(frames, how):
        assert how == "vertical"
        columns = frames[0].columns
        assert all(frame.columns == columns for frame in frames)
        return MiniFrame({c: [v for frame in frames for v in frame.data[c]] for c in columns})


class _Recorder:
    def __init__(self, event, name=None, time_aggregation=None):
        self.kind = type(self).__name__
        self.event, self.name, self.time_aggregation = event, name, time_aggregation


class RemoveTimestepsBeforeEvent(_Recorder):
    pass


class AddCombinedEvent(_Recorder):
    pass


class RemoveTimestepsAfterEvent(_Recorder):
    pass


class _Logger:
    def __init__(self):
        self.messages = []

    def warning(self, message):
        self.messages.append(message)


# --- executed: _json_safe ----------------------------------------------------------------


@pytest.fixture(scope="module")
def json_safe():
    namespace = {
        "__builtins__": {"isinstance": isinstance, "float": float, "dict": dict, "list": list,
                         "str": str, "object": object},
        "math": math,
    }
    _execute([_top_level(_tree(), "_json_safe")], namespace, "pinned_json_safe")
    return namespace["_json_safe"]


def test_json_safe_turns_nonfinite_values_into_null(json_safe):
    value = {"a": float("nan"), "b": [float("inf"), 1.0, {"c": -float("inf")}], 3: 0.0,
             "d": "nan"}
    assert json_safe(value) == {"a": None, "b": [None, 1.0, {"c": None}], "3": 0.0, "d": "nan"}


def _summary(offroad, *, dist=0.4, score=0.8):
    ids = {"run_uuid": "run", "clipgt_id": "clip", "rollout_id": "rollout"}
    metrics = {**ids, "run_name": "synthetic", "progress_clipped_rel": 0.64,
               "progress_rel": 0.64, "gt_dist_traveled_m": 20.0, "collision_at_fault": 0.0,
               "offroad": offroad, "left_corridor_laterally": 0.0,
               "dist_to_gt_trajectory": dist, "lateral_dist_to_gt_trajectory": dist}
    score_metrics = {k: v for k, v in metrics.items() if k not in (*ids, "run_name")}
    score_metrics["progress_score"] = score
    row = {**ids, "run_name": "synthetic", "status": "pass", "passed": True, "score": score,
           "failure_reason": None, "metrics": metrics, "score_metrics": score_metrics}
    return {"schema_version": 1, "scene_score_enabled": True, "rollouts": [row]}


def _normalize(summary, *, metrics_verified=True, full_deviation=0.4):
    identity = {"run_uuid": "run", "clipgt_id": "clip", "rollout_id": "rollout",
                "policy_id": "teacher", "scene_id": "scene", "road_id": "road",
                "stratum": "urban", "replicate_id": 0}
    observation = {"run_uuid": "run", "clipgt_id": "clip", "rollout_id": "rollout",
                   "requested_steps": 200, "actual_steps": 200, "completion_validated": True,
                   "full_max_deviation_m": full_deviation,
                   "required_metrics_verified": metrics_verified,
                   "provenance_ref": "synthetic-fixture-not-real-telemetry"}
    return normalize_summary(summary, [identity], [observation], arm="A", policy_id="teacher",
                             expected_evaluator_sha=SUPPORTED_EVALUATOR_SHA,
                             observed_evaluator_sha=SUPPORTED_EVALUATOR_SHA)


def test_nan_aggregate_reaches_the_summary_as_null_and_normalization_blocks(json_safe):
    # An all-invalid metric aggregates to NaN; the pinned writer serializes it as null.
    written = json.loads(json.dumps(json_safe(_summary(float("nan"))), allow_nan=False))
    assert written["rollouts"][0]["metrics"]["offroad"] is None
    result = _normalize(written)
    assert not result["admitted"]
    assert "missing_required_scoring_metric" in {issue["code"] for issue in result["issues"]}


# --- executed (symbolic): DEFAULT_MODIFIERS ---------------------------------------------


@pytest.fixture(scope="module")
def default_modifiers():
    namespace = {
        "__builtins__": {},
        "pl": PL,
        "RemoveTimestepsBeforeEvent": RemoveTimestepsBeforeEvent,
        "AddCombinedEvent": AddCombinedEvent,
        "RemoveTimestepsAfterEvent": RemoveTimestepsAfterEvent,
    }
    _execute([_top_level(_tree(), "DEFAULT_MODIFIERS")], namespace, "pinned_default_modifiers")
    return namespace["DEFAULT_MODIFIERS"]


def test_default_modifier_list_is_pinned(default_modifiers):
    assert [(m.kind, m.name, m.time_aggregation) for m in default_modifiers] == [
        ("RemoveTimestepsBeforeEvent", None, None),
        ("AddCombinedEvent", "offroad_or_collision", "max"),
        ("AddCombinedEvent", "collision_at_fault", "max"),
        ("AddCombinedEvent", "offroad_or_collision_at_fault", "max"),
        ("AddCombinedEvent", "duration_frac_20s", "last"),
        ("RemoveTimestepsAfterEvent", None, None),
    ]
    assert [m.event.describe() for m in default_modifiers] == [
        "(col(eval_relevant) > 0.0)",
        "((col(collision_any) > 0.0) | (col(offroad) > 0.0))",
        "((col(collision_front) > 0.0) | (col(collision_lateral) > 0))",
        "((col(collision_at_fault) > 0.0) | (col(offroad) > 0.0))",
        "((col(timestamps_us) - col(timestamps_us).min().over(trajectory_uid)) / 20000000.0)",
        "(col(offroad_or_collision) > 0.0)",
    ]


def _apply_events(modifiers, frame):
    """Evaluate each pinned event in order, adding combined columns as float like upstream."""
    events = []
    for modifier in modifiers:
        values = evaluate(modifier.event, frame)
        events.append(values)
        if modifier.kind == "AddCombinedEvent":
            frame = MiniFrame({**frame.data, modifier.name: [float(v) for v in values]})
    return frame, events


def test_warm_up_event_and_early_non_at_fault_contact_known_answers(default_modifiers):
    # Five steps; warm-up until step 2; a contact with no at-fault component at step 3;
    # no map incident. The pinned predicates mark these exact steps.
    frame = MiniFrame({
        "trajectory_uid": ["t"] * 5,
        "timestamps_us": [0, 100_000, 200_000, 300_000, 400_000],
        "eval_relevant": [0.0, 0.0, 1.0, 1.0, 1.0],
        "collision_any": [0.0, 0.0, 0.0, 1.0, 0.0],
        "collision_front": [0.0] * 5,
        "collision_lateral": [0.0] * 5,
        "offroad": [0.0] * 5,
        "dist_to_gt_trajectory": [0.0, 0.0, 0.3, 0.5, 6.0],
    })
    frame, events = _apply_events(default_modifiers, frame)
    warm_up, _, _, _, _, truncation = events
    assert warm_up == [False, False, True, True, True]  # first evaluated step: index 2
    assert frame.data["collision_at_fault"] == [0.0] * 5  # the score never sees this contact
    assert truncation == [False, False, False, True, False]  # first incident: index 3
    # The late 6 m deviation comes after the first incident, so the pinned truncation drops it
    # (test_first_incident_truncation_...); PhysLoop takes the maximum from the full stream.
    assert max(frame.data["dist_to_gt_trajectory"]) == 6.0


def test_at_fault_and_offroad_events_known_answers(default_modifiers):
    frame = MiniFrame({
        "trajectory_uid": ["t"] * 4,
        "timestamps_us": [0, 1, 2, 3],
        "eval_relevant": [1.0] * 4,
        "collision_any": [0.0, 1.0, 1.0, 0.0],
        "collision_front": [0.0, 1.0, 0.0, 0.0],
        "collision_lateral": [0.0, 0.0, 1.0, 0.0],
        "offroad": [0.0, 0.0, 0.0, 1.0],
    })
    frame, _ = _apply_events(default_modifiers, frame)
    assert frame.data["collision_at_fault"] == [0.0, 1.0, 1.0, 0.0]
    assert frame.data["offroad_or_collision"] == [0.0, 1.0, 1.0, 1.0]
    assert frame.data["offroad_or_collision_at_fault"] == [0.0, 1.0, 1.0, 1.0]


def test_duration_fraction_is_per_trajectory_over_twenty_seconds(default_modifiers):
    frame = MiniFrame({
        "trajectory_uid": ["a", "a", "a", "b", "b"],
        "timestamps_us": [1_000_000, 3_000_000, 21_000_000, 5_000_000, 15_000_000],
        "eval_relevant": [1.0] * 5,
        "collision_any": [0.0] * 5,
        "collision_front": [0.0] * 5,
        "collision_lateral": [0.0] * 5,
        "offroad": [0.0] * 5,
    })
    frame, _ = _apply_events(default_modifiers, frame)
    assert frame.data["duration_frac_20s"] == [0.0, 0.1, 1.0, 0.0, 0.5]


# --- executed: pinned modifier row semantics (modifiers.py, main.py) -----------------------


MODIFIER_BUILTINS = {"__build_class__", "bool", "str", "super", "tuple"}
MODIFIER_NODES = ("MetricAggregationModifiers", "remove_timesteps_before_or_after_event",
                  "RemoveTimestepsAfterEvent", "RemoveTimestepsBeforeEvent", "AddCombinedEvent")


@pytest.fixture(scope="module")
def pinned_modifiers():
    tree = _aggregation_tree(MODIFIERS)
    nodes = sorted((_top_level(tree, name) for name in MODIFIER_NODES), key=lambda n: n.lineno)
    # Class statements need __build_class__ and a module __name__; signature annotations
    # evaluate eagerly (bool/str/tuple); AddCombinedEvent.__init__ calls super(). __call__
    # (len, logger, get_removed_rows) is never run, so none of those is provided.
    namespace = {
        "__builtins__": {name: getattr(builtins, name) for name in MODIFIER_BUILTINS},
        "__name__": "pinned_modifiers",
        "ABC": ABC,
        "abstractmethod": abstractmethod,
        "pl": PL,
    }
    return _execute(nodes, namespace, "pinned_modifiers")


@pytest.fixture(scope="module")
def real_default_modifiers(pinned_modifiers):
    namespace = {"__builtins__": {}, "pl": PL,
                 **{name: pinned_modifiers[name] for name in
                    ("RemoveTimestepsBeforeEvent", "AddCombinedEvent",
                     "RemoveTimestepsAfterEvent")}}
    _execute([_top_level(_tree(), "DEFAULT_MODIFIERS")], namespace, "pinned_default_modifiers")
    return namespace["DEFAULT_MODIFIERS"]


def _run_core_modifiers_node(pinned_modifiers, threshold):
    function = _top_level(_aggregation_tree(MAIN), "_run_aggregation_core")
    node, _ = _function_body_statement(
        function, lambda s: isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == "modifiers"
    )
    cfg = SimpleNamespace(aggregation_modifiers=SimpleNamespace(
        max_dist_to_gt_trajectory=threshold))
    namespace = {"__builtins__": {}, "pl": PL, "cfg": cfg,
                 "RemoveTimestepsAfterEvent": pinned_modifiers["RemoveTimestepsAfterEvent"]}
    return _execute([node], namespace, "pinned_main_modifiers")["modifiers"]


@pytest.fixture(scope="module")
def four_metre_modifiers(pinned_modifiers):
    return _run_core_modifiers_node(pinned_modifiers, MAX_DIST_TO_GT_TRAJECTORY_M)


def _apply_chain(modifiers, frame):
    """Mirror processing's ``for modifier in additional_modifiers`` loop on rows only.

    Upstream calls ``modifier(df, agg_df)``; ``__call__`` returns ``self.apply(df)`` for the
    frame plus row-count bookkeeping and the aggregation-table update, neither run here.
    """
    for modifier in modifiers:
        frame = modifier.apply(frame)
    return frame


def test_pinned_modifiers_run_with_an_explicit_builtins_allowlist(pinned_modifiers):
    function = pinned_modifiers["remove_timesteps_before_or_after_event"]
    allowed = function.__globals__["__builtins__"]
    assert type(allowed) is dict
    assert set(allowed) == MODIFIER_BUILTINS
    assert not {"open", "__import__", "eval", "exec", "compile", "getattr"} & set(allowed)
    after = pinned_modifiers["RemoveTimestepsAfterEvent"]
    assert issubclass(after, pinned_modifiers["MetricAggregationModifiers"])


def test_pinned_default_chain_uses_the_pinned_modifier_classes(real_default_modifiers):
    assert [type(m).__name__ for m in real_default_modifiers] == [
        "RemoveTimestepsBeforeEvent", *["AddCombinedEvent"] * 4, "RemoveTimestepsAfterEvent",
    ]
    assert all(type(m).__module__ == "pinned_modifiers" for m in real_default_modifiers)
    assert real_default_modifiers[0].event.describe() == "(col(eval_relevant) > 0.0)"
    assert real_default_modifiers[-1].event.describe() == "(col(offroad_or_collision) > 0.0)"


def test_warm_up_removal_keeps_rows_from_the_first_eval_relevant_step(real_default_modifiers):
    warm_up = real_default_modifiers[0]
    # Unsorted input; trajectory a becomes eval-relevant at t=200 and drops back to 0.0 at
    # t=300; trajectory b is never eval-relevant (its first-event timestamp is null).
    frame = MiniFrame({
        "trajectory_uid": ["a", "b", "a", "a", "b", "a"],
        "timestamps_us": [300, 100, 0, 200, 0, 100],
        "eval_relevant": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    })
    assert warm_up.apply(frame).data == {
        "trajectory_uid": ["a", "a", "b", "b"],
        "timestamps_us": [200, 300, 0, 100],
        "eval_relevant": [1.0, 0.0, 0.0, 0.0],
    }


def test_first_incident_truncation_keeps_the_incident_row_and_drops_later_rows(
        real_default_modifiers):
    # The five-step frame of test_warm_up_event_and_early_non_at_fault_contact_known_answers:
    # warm-up ends at step 2, a non-at-fault contact at step 3, a 6 m deviation at step 4.
    frame = MiniFrame({
        "trajectory_uid": ["t"] * 5,
        "timestamps_us": [0, 100_000, 200_000, 300_000, 400_000],
        "eval_relevant": [0.0, 0.0, 1.0, 1.0, 1.0],
        "collision_any": [0.0, 0.0, 0.0, 1.0, 0.0],
        "collision_front": [0.0] * 5,
        "collision_lateral": [0.0] * 5,
        "offroad": [0.0] * 5,
        "dist_to_gt_trajectory": [0.0, 0.0, 0.3, 0.5, 6.0],
    })
    out = _apply_chain(real_default_modifiers, frame)
    assert out.data["timestamps_us"] == [200_000, 300_000]
    assert out.data["offroad_or_collision"] == [0.0, 1.0]
    assert out.data["collision_at_fault"] == [0.0, 0.0]
    assert out.data["offroad_or_collision_at_fault"] == [0.0, 0.0]
    assert out.data["duration_frac_20s"] == [0.0, 0.005]
    # The 6 m row is gone: the upstream window maximum is 0.5 m, the full stream's is 6.0 m.
    assert out.data["dist_to_gt_trajectory"] == [0.3, 0.5]
    assert "first_event_timestamp" not in out.columns


def test_first_incident_truncation_is_per_trajectory(real_default_modifiers):
    truncation = real_default_modifiers[-1]
    frame = MiniFrame({
        "trajectory_uid": ["a", "a", "a", "b", "b"],
        "timestamps_us": [0, 1, 2, 0, 1],
        "offroad_or_collision": [0.0, 1.0, 1.0, 0.0, 0.0],
    })
    assert truncation.apply(frame).data == {
        "trajectory_uid": ["a", "a", "b", "b"],
        "timestamps_us": [0, 1, 0, 1],
        "offroad_or_collision": [0.0, 1.0, 0.0, 0.0],
    }


def test_main_builds_exactly_one_four_metre_truncation_modifier(four_metre_modifiers,
                                                                 pinned_modifiers):
    [modifier] = four_metre_modifiers
    assert type(modifier) is pinned_modifiers["RemoveTimestepsAfterEvent"]
    assert modifier.event.describe() == "(col(dist_to_gt_trajectory) >= 4.0)"
    # main passes this list to processing as additional_modifiers, after DEFAULT_MODIFIERS.
    function = _top_level(_aggregation_tree(MAIN), "_aggregate_metrics")
    [call] = [n for n in ast.walk(function) if isinstance(n, ast.Call)
              and ast.unparse(n.func) == "processing.aggregate_and_write_metrics_results_txt"]
    assert {k.arg: ast.unparse(k.value) for k in call.keywords}["additional_modifiers"] == (
        "modifiers")


def test_four_metre_modifier_keeps_the_first_crossing_and_drops_later_rows(
        four_metre_modifiers):
    [modifier] = four_metre_modifiers
    frame = MiniFrame({
        "trajectory_uid": ["a"] * 5 + ["b"] * 2,
        "timestamps_us": [0, 1, 2, 3, 4, 0, 1],
        # a: 3.999 does not trigger, exactly 4.0 does; b: a null sample and 3.99 never do.
        "dist_to_gt_trajectory": [0.3, 3.999, 4.0, 6.0, 1.0, None, 3.99],
    })
    out = modifier.apply(frame)
    assert out.data == {
        "trajectory_uid": ["a", "a", "a", "b", "b"],
        "timestamps_us": [0, 1, 2, 0, 1],
        "dist_to_gt_trajectory": [0.3, 3.999, 4.0, None, 3.99],
    }
    # The crossing row survives, so the upstream window maximum for a is 4.0, not 6.0.
    assert max(out.data["dist_to_gt_trajectory"][:3]) == 4.0


def test_upstream_order_cuts_at_four_metres_before_a_later_collision(
        real_default_modifiers, four_metre_modifiers):
    frame = MiniFrame({
        "trajectory_uid": ["t"] * 6,
        "timestamps_us": [0, 100, 200, 300, 400, 500],
        "eval_relevant": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "collision_any": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "collision_front": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "collision_lateral": [0.0] * 6,
        "offroad": [0.0] * 6,
        "dist_to_gt_trajectory": [0.0, 0.1, 1.0, 4.5, 7.0, 8.0],
    })
    # Default chain: warm-up drops t=0, the at-fault collision keeps t<=400; then the >= 4 m
    # modifier keeps t<=300. The collision row never reaches the summary aggregation.
    out = _apply_chain([*real_default_modifiers, *four_metre_modifiers], frame)
    assert out.data["timestamps_us"] == [100, 200, 300]
    assert out.data["dist_to_gt_trajectory"] == [0.1, 1.0, 4.5]
    assert out.data["collision_at_fault"] == [0.0, 0.0, 0.0]
    assert out.data["offroad_or_collision"] == [0.0, 0.0, 0.0]
    default_only = _apply_chain(real_default_modifiers, frame)
    assert default_only.data["timestamps_us"] == [100, 200, 300, 400]
    assert default_only.data["collision_at_fault"] == [0.0, 0.0, 0.0, 1.0]


# --- executed (partial): missing-offroad zero-fill -----------------------------------------


def _zero_fill_node():
    function = _top_level(_tree(), "aggregate_and_write_metrics_results_txt")

    def is_zero_fill(statement):
        test = getattr(statement, "test", None)
        return (
            isinstance(statement, ast.If)
            and isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Constant)
            and test.left.value == "offroad"
            and isinstance(test.ops[0], ast.NotIn)
        )

    return _function_body_statement(function, is_zero_fill)


def _run_zero_fill(frame):
    node, _ = _zero_fill_node()
    namespace = {"__builtins__": {}, "df_wide": frame, "pl": PL, "logger": _Logger(),
                 "processing_warnings": []}
    _execute([node], namespace, "pinned_offroad_zero_fill")
    return namespace


def test_upstream_zero_fills_a_missing_offroad_metric():
    frame = MiniFrame({"trajectory_uid": ["t", "t"], "timestamps_us": [0, 1],
                       "collision_any": [0.0, 0.0]})
    namespace = _run_zero_fill(frame)
    assert namespace["df_wide"].data["offroad"] == [0.0, 0.0]
    message = ("No offroad column found in the metrics dataframe. "
               "Adding a default value of 0.0.")
    assert namespace["processing_warnings"] == [message]
    assert namespace["logger"].messages == [message]


def test_present_offroad_metric_is_left_unchanged():
    frame = MiniFrame({"trajectory_uid": ["t"], "timestamps_us": [0], "offroad": [1.0]})
    namespace = _run_zero_fill(frame)
    assert namespace["df_wide"] is frame
    assert namespace["processing_warnings"] == []


def test_zero_filled_offroad_feeds_the_modifiers_and_the_scorer():
    function = _top_level(_tree(), "aggregate_and_write_metrics_results_txt")
    _, fill = _zero_fill_node()
    statements = [ast.unparse(statement) for statement in function.body]
    loop = next(i for i, s in enumerate(statements) if s.startswith("for modifier in "))
    assert fill < loop
    assert "additional_modifiers = DEFAULT_MODIFIERS + (additional_modifiers or [])" in (
        statements
    )


def test_physloop_keeps_normalization_blocked_when_upstream_zero_filled_offroad():
    # The summary reads offroad == 0.0, but the raw metric was absent (map not loaded).
    result = _normalize(_summary(0.0), metrics_verified=False)
    assert not result["admitted"]
    assert result["records"] == []
    assert "metrics_unverified" in {issue["code"] for issue in result["issues"]}
    assert _normalize(_summary(0.0), metrics_verified=True)["admitted"]


# --- executed: _add_progress_clipped_rel_metric -------------------------------------------


@pytest.fixture(scope="module")
def add_progress_clipped():
    namespace = {"__builtins__": {"tuple": tuple}, "pl": PL}
    _execute([_top_level(_tree(), "_add_progress_clipped_rel_metric")], namespace,
             "pinned_progress_clipped")
    return namespace["_add_progress_clipped_rel_metric"]


def test_progress_clipped_is_the_last_progress_of_the_modified_window(add_progress_clipped):
    frame = MiniFrame({"progress_rel_to_total": [0.1, 0.4]})
    aggregations = MiniFrame({"name": ["progress_rel_to_total"], "time_aggregation": ["last"]})
    frame_out, aggregations_out = add_progress_clipped(frame, aggregations)
    assert frame_out.data["progress_clipped_rel"] == [0.1, 0.4]
    assert aggregations_out.data == {
        "name": ["progress_rel_to_total", "progress_clipped_rel"],
        "time_aggregation": ["last", "last"],
    }


def test_progress_clipped_without_source_or_already_present_is_unchanged(add_progress_clipped):
    aggregations = MiniFrame({"name": [], "time_aggregation": []})
    absent = MiniFrame({"offroad": [0.0]})
    assert add_progress_clipped(absent, aggregations) == (absent, aggregations)
    present = MiniFrame({"progress_rel_to_total": [0.2], "progress_clipped_rel": [0.9]})
    assert add_progress_clipped(present, aggregations) == (present, aggregations)


def test_progress_clipped_is_added_after_all_modifiers():
    function = _top_level(_tree(), "aggregate_and_write_metrics_results_txt")
    statements = [ast.unparse(statement) for statement in function.body]
    loop = next(i for i, s in enumerate(statements) if s.startswith("for modifier in "))
    clipped = next(i for i, s in enumerate(statements) if "_add_progress_clipped_rel_metric(" in s)
    aggregated = next(i for i, s in enumerate(statements) if s.startswith("df_wide_avg_t = "))
    assert loop < clipped < aggregated
    assert statements[aggregated].startswith("df_wide_avg_t = df_wide_modified.group_by(")


# --- invalid frames: writer aggregation versus the Polars processing path ------------------


METRIC_RETURN_BUILTINS = {"__build_class__", "bool", "float", "int", "len", "list", "str",
                          "ValueError"}


@pytest.fixture(scope="module")
def metric_return():
    tree = ast.parse(_source(DATA, DATA_SHA256), filename=DATA)
    nodes = [node for node in tree.body
             if getattr(node, "name", None) in {"AggregationType", "MetricReturn"}]
    assert {node.name for node in nodes} == {"AggregationType", "MetricReturn"}
    # Class statements need __build_class__ and a module __name__; annotations evaluate at
    # class creation (list/int/float/bool/str) and method bodies use float/len/ValueError.
    # ValueError serves only the pinned "unknown aggregation type" branch.
    namespace = {
        "__builtins__": {name: getattr(builtins, name) for name in METRIC_RETURN_BUILTINS},
        "__name__": "pinned_metric_return",
        "dataclasses": dataclasses,
        "np": np,
        "StrEnum": StrEnum,
        "pl": PL,
        "Callable": Callable,
    }
    _execute(nodes, namespace, "pinned_metric_return")
    return namespace["MetricReturn"], namespace["AggregationType"]


def test_metric_return_runs_with_an_explicit_builtins_allowlist(metric_return):
    # Build-spec section 2 rule 9 (review finding A7-04): pinned class bodies execute with a
    # restricted namespace, never with the full builtins exec injects by default.
    metric, kind = metric_return
    for function in (metric.aggregate, kind.get_numpy_func):
        allowed = function.__globals__["__builtins__"]
        assert type(allowed) is dict
        assert set(allowed) == METRIC_RETURN_BUILTINS
        assert not {"open", "__import__", "eval", "exec", "compile", "getattr"} & set(allowed)


def test_writer_aggregate_excludes_invalid_samples(metric_return):
    metric, kind = metric_return

    def aggregate(values, valid, mode):
        return metric(name="m", timestamps_us=list(range(len(values))), values=values,
                      valid=valid, time_aggregation=kind(mode)).aggregate()

    assert aggregate([0.0, 9.0], [True, False], "max") == 0.0
    assert aggregate([1.0, 2.0, 3.0], [True, True, False], "last") == 2.0
    assert math.isnan(aggregate([5.0], [False], "max"))


def test_processing_pivot_reads_values_and_never_the_validity_mask():
    tree = _tree()
    function = _top_level(tree, "aggregate_and_write_metrics_results_txt")
    pivot, _ = _function_body_statement(
        function, lambda s: ast.unparse(s).startswith("df_wide = df_long.pivot(")
    )
    call = pivot.value.func.value  # df_long.pivot(...).sort(...)
    keywords = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
    assert keywords == {"values": "values", "index": ["trajectory_uid", "timestamps_us"],
                        "on": "name"}
    # No "valid" literal anywhere in the module: an invalid sample's value is aggregated
    # like a valid one on this path (contrast with MetricReturn.aggregate above).
    assert not any(isinstance(n, ast.Constant) and n.value == "valid" for n in ast.walk(tree))


# --- structural: scorer call site ------------------------------------------------------


def test_scorer_call_site_is_pinned():
    tree = _tree()
    writer = _top_level(tree, "write_results_summary_json")
    calls = [n for n in ast.walk(writer)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "score_rollout"]
    assert len(calls) == 1
    assert [ast.unparse(arg) for arg in calls[0].args] == ["row", "scene_score_config"]
    assert calls[0].keywords == []
    loop = next(n for n in ast.walk(writer) if isinstance(n, ast.For)
                and ast.unparse(n.iter) == "df_rollouts.to_dicts()")
    assert any(call is node for node in ast.walk(loop) for call in calls)
    assert "df_rollouts = df_wide_avg_t.sort(sort_columns) if sort_columns else df_wide_avg_t" in {
        ast.unparse(s) for s in writer.body
    }
    function = _top_level(tree, "aggregate_and_write_metrics_results_txt")
    writes = [n for n in ast.walk(function) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Name) and n.func.id == "write_results_summary_json"]
    assert len(writes) == 1
    assert ast.unparse(writes[0].args[0]) == "df_wide_avg_t"


def test_failed_rollout_rows_are_appended_by_the_writer():
    writer = _top_level(_tree(), "write_results_summary_json")
    source = ast.unparse(writer)
    assert "rollout_rows.extend(failed_rollout_summary_rows(failed_rollouts, " in source
