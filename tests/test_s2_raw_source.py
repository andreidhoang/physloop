"""Independent source receipts and literal writer execution for S2 raw fixtures."""

import ast
import dataclasses
import hashlib
import json
from enum import StrEnum
from pathlib import Path

import numpy as np

from gate.s2_extract import PARQUET_COLUMNS, RAW_AGGREGATIONS

ROOT = Path(__file__).parent / "fixtures" / "alpasim_s2_raw"
# Every vendored file: four scorer/writer snapshots, processing.py and the three
# aggregation sources CPU-12 executes (modifiers.py, main.py, scene_score.py).
SOURCE_PATHS = {
    "data_e121e591.py.txt": "src/eval/src/eval/data.py",
    "processing_e121e591.py.txt": "src/eval/src/eval/aggregation/processing.py",
    "ground_truth_e121e591.py.txt": "src/eval/src/eval/scorers/ground_truth.py",
    "collision_e121e591.py.txt": "src/eval/src/eval/scorers/collision.py",
    "offroad_e121e591.py.txt": "src/eval/src/eval/scorers/offroad.py",
    "image_e121e591.py.txt": "src/eval/src/eval/scorers/image.py",
    "modifiers_e121e591.py.txt": "src/eval/src/eval/aggregation/modifiers.py",
    "main_e121e591.py.txt": "src/eval/src/eval/aggregation/main.py",
    "scene_score_e121e591.py.txt": "src/eval/src/eval/aggregation/scene_score.py",
}


def test_complete_public_source_snapshots_match_git_and_sha256_receipts():
    provenance = json.loads((ROOT / "provenance.json").read_text())
    assert provenance["commit"] == "e121e59141dc391f6c295513d6dadc840a419149"
    assert len(provenance["sources"]) == 9
    assert {e["file"]: e["path"] for e in provenance["sources"]} == SOURCE_PATHS
    for entry in [*provenance["sources"], provenance["license_file"]]:
        raw = (ROOT / entry["file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        blob = b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
        assert hashlib.sha1(blob).hexdigest() == entry["git_blob_sha"]
        assert provenance["commit"] in entry["source_url"]


class _WriterBoundary:
    """Observe the real writer's Polars boundary without importing all AlpaSim."""

    @staticmethod
    def from_dicts(dictionaries):
        class Frame:
            def explode(self, columns):
                return dictionaries, columns

        return Frame()


def _writer_namespace():
    source = ast.parse((ROOT / "data_e121e591.py.txt").read_text())
    wanted = {"MetricReturn", "AggregationType", "create_metrics_dataframe"}
    nodes = [node for node in source.body if getattr(node, "name", None) in wanted]
    assert {node.name for node in nodes} == wanted
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
              *nodes],
        type_ignores=[],
    )
    namespace = {"dataclasses": dataclasses, "np": np, "StrEnum": StrEnum, "pl": _WriterBoundary}
    # Execute only the three audited definitions from content-addressed public source.
    exec(compile(ast.fix_missing_locations(module), "pinned_data_writer", "exec"), namespace)  # noqa: S102
    return namespace


def test_actual_pinned_writer_casts_values_and_preserves_invalid_samples():
    namespace = _writer_namespace()
    metric = namespace["MetricReturn"](
        name="collision_front", timestamps_us=[100, 200], values=[False, True],
        valid=[True, False], time_aggregation=namespace["AggregationType"].MAX,
        info="Not a serialized parquet column",
    )
    dictionaries, explode_columns = namespace["create_metrics_dataframe"](
        [metric], "clip", "rollout", "run", "bf16"
    )
    assert explode_columns == ["values", "timestamps_us", "valid"]
    assert dictionaries == [{
        "name": "collision_front", "timestamps_us": [100, 200], "values": [0.0, 1.0],
        "valid": [True, False], "time_aggregation": "max", "clipgt_id": "clip",
        "rollout_id": "rollout", "run_uuid": "run", "run_name": "bf16",
    }]
    assert all(type(value) is float for value in dictionaries[0]["values"])
    assert set(dictionaries[0]) == PARQUET_COLUMNS


def test_extractor_required_names_and_aggregations_exist_in_pinned_raw_scorers():
    enum = _writer_namespace()["AggregationType"]
    emitted = {}
    for filename in ("ground_truth", "collision", "offroad", "image"):
        source = ast.parse((ROOT / f"{filename}_e121e591.py.txt").read_text())
        for node in ast.walk(source):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "MetricReturn":
                continue
            kwargs = {item.arg: item.value for item in node.keywords}
            name, aggregation = kwargs["name"], kwargs["time_aggregation"]
            assert isinstance(name, ast.Constant)
            assert isinstance(aggregation, ast.Attribute)
            emitted[name.value] = enum[aggregation.attr].value
    assert all(emitted.get(name) == mode for name, mode in RAW_AGGREGATIONS.items())
    # Both fields exist only after aggregation; requiring them in raw parquet is a bug.
    assert "progress_clipped_rel" not in emitted
    assert "collision_at_fault" not in emitted
    assert emitted["progress_rel_to_total"] == "last"
    assert emitted["collision_front"] == emitted["collision_lateral"] == "max"
