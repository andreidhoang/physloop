"""CPU-03 known answers for the S2 population and contamination audit.

Literals are hand-derived from the readiness specification (CPU-03) and the
synthetic fixture in ``tests/fixtures/s2_analysis/population_manifest.json``.
No licensed metadata, network, provider CLI or GPU tool is used.
"""

import json
import random
from copy import deepcopy
from pathlib import Path

import pytest

from gate.s2_identity import identity_hash
from gate.s2_population import ROLES, audit_population, half_open_overlap

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "s2_analysis" / "population_manifest.json").read_text()
)
RULE = {
    "schema_version": 1,
    "rule_id": "synthetic-rule",
    "source_ref": "test",
    "separated_roles": [["pilot", "confirmation"]],
    "dimensions": ["road", "temporal"],
}


def manifest():
    return deepcopy(FIXTURE["manifest"])


def roles():
    return deepcopy(FIXTURE["roles"])


def scene(scene_id, road, stratum="urban", **changes):
    return {
        "scene_id": scene_id,
        "asset_sha256": identity_hash(scene_id),
        "road_id": road,
        "road_evidence_ref": f"synthetic:{road}",
        "stratum": stratum,
        "split_label": None,
        "source": {"capture_id": None, "clip_id": None, "interval": None},
        **changes,
    }


def wrap(scenes):
    return {
        "artifact_kind": "s2_population_manifest",
        "schema_version": 1,
        "scope": "synthetic",
        "catalog_revision": "r0",
        "scenes": scenes,
    }


def codes(report):
    return {item["code"] for item in report["issues"]}


# --- interval arithmetic -------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ((0, 10), (8, 12), 2),
        ((8, 12), (0, 10), 2),
        ((0, 10), (10, 12), 0),
        ((10, 12), (0, 10), 0),
        ((0, 10), (2, 3), 1),
        ((0, 10), (0, 10), 10),
        ((0, 1), (5, 6), 0),
        ((0.5, 9.5), (9.0, 12.0), 0.5),
    ],
)
def test_half_open_overlap_known_answers(a, b, expected):
    assert half_open_overlap(*a, *b) == expected


# --- fixture known answers -----------------------------------------------------------


def test_fixture_counts_and_overlaps():
    report = audit_population(manifest(), roles())
    expected = FIXTURE["expected"]
    counts = report["counts"]
    assert report["status"] == "passed", report["issues"]
    assert counts["scenes"] == expected["scene_count"] == 4
    assert counts["roads"] == expected["road_count"] == 2
    assert counts["scenes_by_road"] == expected["scenes_by_road"]
    assert counts["roads_by_stratum"] == expected["roads_by_stratum"]
    assert counts["split_label_counts"] == expected["split_label_counts"]
    assert counts["unlabelled_scenes"] == expected["unlabelled_scenes"] == 1
    overlaps = [
        {key: row[key] for key in ("scene_ids", "capture_id", "unit", "overlap")}
        for row in report["temporal_overlaps"]
    ]
    assert overlaps == expected["temporal_overlaps"]
    assert (
        report["lineage"]["temporal_unverified_scene_ids"]
        == (expected["temporal_unverified_scene_ids"])
    )
    intersections = {
        "|".join(row["roles"]): row for row in report["role_membership"]["intersections"]
    }
    assert set(intersections) == set(expected["intersections"])
    for name, want in expected["intersections"].items():
        row = intersections[name]
        for dimension in ("scene", "road", "clip", "temporal"):
            assert row["dimensions"][dimension]["status"] == want[dimension], (name, dimension)
        assert row["dimensions"]["road"]["shared"] == want["shared_roads"]


def test_zero_overlap_at_half_open_boundary_is_not_reported():
    report = audit_population(manifest(), roles())
    pairs = [row["scene_ids"] for row in report["temporal_overlaps"]]
    assert ["scene-a", "scene-c"] not in pairs


def test_float_intervals_overlap_in_report():
    base = {"capture_id": "cap", "clip_id": None}
    scenes = [
        scene(
            "s1",
            "road-1",
            source={
                **base,
                "interval": {"start": 0.5, "end": 9.5, "unit": "s", "convention": "half_open"},
            },
        ),
        scene(
            "s2",
            "road-1",
            source={
                **base,
                "interval": {"start": 9.0, "end": 12.0, "unit": "s", "convention": "half_open"},
            },
        ),
    ]
    report = audit_population(wrap(scenes))
    assert [row["overlap"] for row in report["temporal_overlaps"]] == [0.5]


def test_a_real_label_named_unlabelled_does_not_collide():
    report = audit_population(
        wrap([scene("s1", "road-1", split_label="unlabelled"), scene("s2", "road-1")])
    )
    assert report["counts"]["split_label_counts"] == {"unlabelled": 1}
    assert report["counts"]["unlabelled_scenes"] == 1


def test_split_label_is_preserved_and_never_contamination():
    report = audit_population(manifest(), roles())
    labels = {row["scene_id"]: row["split_label"] for row in report["scenes"]}
    assert labels == {
        "scene-a": "alpamayo_aligned",
        "scene-b": "alpamayo_aligned",
        "scene-c": None,
        "scene-d": "alpamayo_aligned",
    }
    assert not any("alpamayo" in item["code"] for item in report["issues"])
    assert not any("split" in item["code"] for item in report["issues"])


def test_missing_capture_lineage_is_unverified_not_clean():
    report = audit_population(manifest(), roles())
    assert report["lineage"]["missing"]["capture_id"] == 1
    assert report["lineage"]["missing"]["clip_id"] == 1
    assert report["lineage"]["missing"]["interval"] == 1
    assert "scene-d" in report["lineage"]["temporal_unverified_scene_ids"]


# --- overlap rule ------------------------------------------------------------------


def test_facts_are_computed_but_decisions_blocked_without_rule():
    report = audit_population(manifest(), roles())
    assert report["decisions"]["status"] == "blocked"
    assert report["decisions"]["code"] == "overlap_rule_unregistered"
    assert report["decisions"]["pairs"] == []
    assert len(report["temporal_overlaps"]) == 2
    assert report["status"] == "passed"


def test_supplied_rule_reports_violation():
    report = audit_population(manifest(), roles(), overlap_rule=RULE)
    decisions = report["decisions"]
    assert decisions["status"] == "blocked"
    assert decisions["code"] == "overlap_rule_violated"
    assert decisions["rule"]["status"] == "supplied_not_authenticated"
    assert decisions["rule"]["sha256"] == identity_hash(RULE)
    pair = decisions["pairs"][0]
    assert pair["roles"] == ["confirmation", "pilot"]
    assert pair["decision"] == "violated"
    assert pair["dimensions"] == {"road": "violated", "temporal": "violated"}


def test_rule_with_unknown_lineage_is_unverified():
    rule = {**RULE, "separated_roles": [["development", "pilot"]], "dimensions": ["clip"]}
    report = audit_population(manifest(), roles(), overlap_rule=rule)
    assert report["decisions"]["status"] == "unverified"
    assert report["decisions"]["code"] == "overlap_unverified"
    assert report["decisions"]["pairs"][0]["decision"] == "unverified"


def test_rule_on_known_disjoint_dimension_is_separated():
    rule = {**RULE, "separated_roles": [["development", "pilot"]], "dimensions": ["road", "scene"]}
    report = audit_population(manifest(), roles(), overlap_rule=rule)
    assert report["decisions"]["status"] == "passed"
    assert report["decisions"]["pairs"][0]["decision"] == "separated"


@pytest.mark.parametrize(
    "rule",
    [
        {**RULE, "schema_version": 2},
        {**RULE, "dimensions": []},
        {**RULE, "dimensions": ["roads"]},
        {**RULE, "separated_roles": [["pilot", "pilot"]]},
        {**RULE, "separated_roles": [["pilot", "training"]]},
        {**RULE, "separated_roles": []},
        {**RULE, "min_overlap": 1},
        {key: value for key, value in RULE.items() if key != "source_ref"},
    ],
)
def test_malformed_rule_blocks_decisions(rule):
    report = audit_population(manifest(), roles(), overlap_rule=rule)
    assert report["decisions"]["status"] == "blocked"
    assert report["decisions"]["code"] == "invalid_overlap_rule"
    assert "invalid_overlap_rule" in codes(report)
    assert len(report["temporal_overlaps"]) == 2


def test_rule_without_roles_is_blocked():
    report = audit_population(manifest(), None, overlap_rule=RULE)
    assert report["decisions"]["code"] == "roles_not_supplied"
    assert report["role_membership"]["status"] == "not_run"


# --- rejections ---------------------------------------------------------------------


def test_road_assigned_two_strata_rejects():
    data = wrap([scene("s1", "road-1", "urban"), scene("s2", "road-1", "highway")])
    report = audit_population(data)
    assert report["status"] == "blocked"
    assert "road_stratum_conflict" in codes(report)
    assert report["counts"]["roads"] == 1
    assert report["counts"]["road_stratum_conflicts"] == ["road-1"]


def test_changed_asset_under_same_id_rejects():
    first = scene("s1", "road-1")
    changed = {**first, "asset_sha256": "f" * 64}
    report = audit_population(wrap([first, changed, scene("s2", "road-2")]))
    assert report["status"] == "blocked"
    assert "asset_changed_under_same_id" in codes(report)
    assert report["counts"]["conflicting_scene_ids"] == ["s1"]
    assert report["counts"]["scenes"] == 1


def test_other_change_under_same_id_rejects():
    first = scene("s1", "road-1")
    changed = {**first, "road_id": "road-9"}
    report = audit_population(wrap([first, changed]))
    assert "scene_record_conflict" in codes(report)
    assert report["status"] == "blocked"


def test_duplicate_content_under_different_ids_rejects():
    first = scene("s1", "road-1")
    twin = {**scene("s2", "road-1"), "asset_sha256": first["asset_sha256"]}
    report = audit_population(wrap([first, twin]))
    assert "duplicate_content_different_ids" in codes(report)
    assert report["status"] == "blocked"
    assert report["counts"]["duplicate_content"] == [
        {"asset_sha256": first["asset_sha256"], "scene_ids": ["s1", "s2"]}
    ]


def test_missing_asset_hash_is_unverified_not_rejected():
    report = audit_population(wrap([scene("s1", "road-1", asset_sha256=None)]))
    assert report["status"] == "passed"
    assert report["lineage"]["missing"]["asset_sha256"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"scene_id": ""},
        {"asset_sha256": "xyz"},
        {"road_id": 7},
        {"source": {"capture_id": "c", "clip_id": "k"}},
        {
            "source": {
                "capture_id": "c",
                "clip_id": "k",
                "interval": {"start": 5, "end": 5, "unit": "us", "convention": "half_open"},
            }
        },
        {
            "source": {
                "capture_id": "c",
                "clip_id": "k",
                "interval": {"start": 0, "end": True, "unit": "us", "convention": "half_open"},
            }
        },
        {"extra": 1},
    ],
)
def test_malformed_scene_row_rejects(change):
    report = audit_population(wrap([{**scene("s1", "road-1"), **change}, scene("s2", "road-2")]))
    assert report["status"] == "blocked"
    assert "invalid_scene_record" in codes(report)
    assert report["counts"]["invalid_scene_records"] == 1


def test_unsupported_convention_and_unit_mismatch_are_unverified():
    base = {"capture_id": "cap", "clip_id": None}
    scenes = [
        scene(
            "s1",
            "road-1",
            source={
                **base,
                "interval": {"start": 0, "end": 10, "unit": "us", "convention": "half_open"},
            },
        ),
        scene(
            "s2",
            "road-1",
            source={
                **base,
                "interval": {"start": 0, "end": 10, "unit": "ms", "convention": "half_open"},
            },
        ),
        scene(
            "s3",
            "road-1",
            source={
                **base,
                "interval": {"start": 0, "end": 10, "unit": "us", "convention": "closed"},
            },
        ),
    ]
    report = audit_population(wrap(scenes))
    assert "unsupported_interval_convention" in codes(report)
    assert report["temporal_overlaps"] == []
    assert report["temporal_unverified_pairs"] == [
        {"scene_ids": ["s1", "s2"], "capture_id": "cap", "reason": "interval_unit_mismatch"}
    ]
    assert report["lineage"]["temporal_unverified_scene_ids"] == ["s3"]


# --- roles ---------------------------------------------------------------------------


def test_role_intersection_on_the_same_scene_is_reported():
    membership = roles()
    membership["development"].append("scene-a")
    report = audit_population(manifest(), membership)
    rows = {"|".join(row["roles"]): row for row in report["role_membership"]["intersections"]}
    scene_dimension = rows["development|pilot"]["dimensions"]["scene"]
    assert scene_dimension == {"status": "overlapping", "shared": ["scene-a"]}


@pytest.mark.parametrize(
    "membership,code",
    [
        ({"training": ["scene-a"]}, "unknown_role"),
        ({"pilot": ["scene-z"]}, "role_scene_unknown"),
        ({"pilot": [""]}, "invalid_role_membership"),
        ({"pilot": "scene-a"}, "invalid_role_membership"),
    ],
)
def test_role_negative_controls(membership, code):
    report = audit_population(manifest(), membership)
    assert code in codes(report)
    assert report["status"] == "blocked"


def test_role_vocabulary():
    assert ROLES == ("confirmation", "development", "instrument", "pilot")


# --- replicate independence and reorder invariance ----------------------------------


def test_three_scenes_on_two_roads_remain_two_roads_across_replicates():
    scenes = [scene("s1", "road-1"), scene("s2", "road-1"), scene("s3", "road-2")]
    once = audit_population(wrap(scenes), {"pilot": ["s1", "s2", "s3"]})
    # A per-job (replicate-expanded) listing repeats identical scene rows and role entries.
    thrice = audit_population(wrap(scenes * 3), {"pilot": ["s1", "s2", "s3"] * 3})
    for report in (once, thrice):
        assert report["counts"]["scenes"] == 3
        assert report["counts"]["roads"] == 2
        assert report["counts"]["scenes_by_road"] == {"road-1": 2, "road-2": 1}
    assert once["counts"] == {**thrice["counts"], "duplicate_scene_records": 0}
    assert thrice["counts"]["duplicate_scene_records"] == 6
    assert once["scenes"] == thrice["scenes"]
    assert once["role_membership"]["roles"] == thrice["role_membership"]["roles"]
    assert thrice["status"] == "passed"


def test_reordered_inputs_give_identical_canonical_report():
    baseline = audit_population(manifest(), roles(), overlap_rule=RULE)
    shuffled = manifest()
    random.Random(3).shuffle(shuffled["scenes"])
    membership = {name: list(reversed(ids)) for name, ids in reversed(list(roles().items()))}
    again = audit_population(shuffled, membership, overlap_rule=deepcopy(RULE))
    assert identity_hash(again) == identity_hash(baseline)


# --- artifact contract ---------------------------------------------------------------


def test_artifact_contract_and_inputs_unchanged():
    data, membership = manifest(), roles()
    original = deepcopy((data, membership))
    report = audit_population(data, membership)
    assert (data, membership) == original
    assert report["artifact_kind"] == "s2_population_audit"
    assert report["schema_version"] == 1
    assert report["scope"] == "synthetic"
    assert report["producer"]["module"] == "gate.s2_population"
    assert len(report["producer"]["source_sha256"]) == 64
    assert set(report["input_hashes"]) == {"manifest", "roles", "overlap_rule"}
    assert report["input_hashes"]["overlap_rule"] is None
    assert report["claims"]["road_independence"] == "not_established"
    assert report["claims"]["unseen_training_data_cleanliness"] == "not_established"
    for item in report["issues"]:
        assert set(item) == {"code", "path", "severity", "evidence_refs", "message"}
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.update(schema_version=2),
        lambda m: m.update(artifact_kind="other"),
        lambda m: m.update(scope="real"),
        lambda m: m.pop("catalog_revision"),
        lambda m: m.update(scenes={}),
        lambda m: m.update(extra=True),
    ],
)
def test_manifest_envelope_fails_closed(mutate):
    data = manifest()
    mutate(data)
    with pytest.raises(ValueError):
        audit_population(data)


def test_roles_must_be_mapping_or_none():
    with pytest.raises(ValueError):
        audit_population(manifest(), [["pilot", "scene-a"]])


def test_non_json_manifest_fails_closed():
    data = manifest()
    data["scenes"][0]["source"]["interval"]["end"] = float("nan")
    with pytest.raises(ValueError):
        audit_population(data)


# --- review regressions (A5-F1, F2, F6, F10, P2) -------------------------------------


def timed(scene_id, road, start, end, capture="cap", **changes):
    return scene(
        scene_id,
        road,
        source={
            "capture_id": capture,
            "clip_id": f"clip-{scene_id}",
            "interval": {"start": start, "end": end, "unit": "us", "convention": "half_open"},
        },
        **changes,
    )


def rule(dimensions, pair=("pilot", "confirmation")):
    return {**RULE, "separated_roles": [list(pair)], "dimensions": dimensions}


def test_shared_scene_is_temporal_overlap_under_temporal_only_rule():
    data = wrap([timed("s1", "road-1", 0, 10), timed("s2", "road-2", 50, 60)])
    report = audit_population(
        data, {"pilot": ["s1"], "confirmation": ["s1", "s2"]}, overlap_rule=rule(["temporal"])
    )
    facts = report["role_membership"]["intersections"][0]["dimensions"]
    assert facts["temporal"] == {"status": "overlapping", "shared": [["s1", "s1"]]}
    assert report["decisions"]["pairs"][0]["decision"] == "violated"
    assert report["decisions"]["status"] == "blocked"
    assert report["decisions"]["code"] == "overlap_rule_violated"


def test_shared_scene_without_interval_is_temporal_unverified():
    data = wrap([scene("s1", "road-1"), timed("s2", "road-2", 50, 60)])
    report = audit_population(
        data, {"pilot": ["s1"], "confirmation": ["s1", "s2"]}, overlap_rule=rule(["temporal"])
    )
    facts = report["role_membership"]["intersections"][0]["dimensions"]
    assert facts["temporal"]["status"] == "unverified"
    assert report["decisions"]["status"] == "unverified"


def separated_manifest():
    return wrap([timed("s1", "road-1", 0, 10), timed("s2", "road-2", 50, 60)])


ALL_DIMENSIONS = ["clip", "road", "scene", "temporal"]


def test_clean_separation_control_passes():
    report = audit_population(
        separated_manifest(),
        {"pilot": ["s1"], "confirmation": ["s2"]},
        overlap_rule=rule(ALL_DIMENSIONS),
    )
    assert report["decisions"]["status"] == "passed"
    assert report["role_membership"]["status"] == "passed"


def test_unknown_role_member_makes_separation_unverified():
    report = audit_population(
        separated_manifest(),
        {"pilot": ["s1", "ghost"], "confirmation": ["s2"]},
        overlap_rule=rule(ALL_DIMENSIONS),
    )
    assert report["status"] == "blocked"
    assert report["role_membership"]["status"] == "blocked"
    pair = report["decisions"]["pairs"][0]
    assert pair["decision"] == "unverified"
    assert pair["incomplete_roles"] == ["pilot"]
    assert report["decisions"]["status"] != "passed"
    assert report["role_membership"]["roles"]["pilot"]["membership_complete"] is False


def test_dropped_member_does_not_hide_a_known_violation():
    data = wrap([timed("s1", "road-1", 0, 10), timed("s2", "road-1", 50, 60)])
    report = audit_population(
        data, {"pilot": ["s1", "ghost"], "confirmation": ["s2"]}, overlap_rule=rule(["road"])
    )
    assert report["decisions"]["pairs"][0]["decision"] == "violated"
    assert report["decisions"]["code"] == "overlap_rule_violated"


def test_empty_role_is_unverified_not_vacuously_separated():
    report = audit_population(
        separated_manifest(),
        {"pilot": [], "confirmation": ["s1"]},
        overlap_rule=rule(["road", "scene"]),
    )
    assert report["decisions"]["pairs"][0]["decision"] == "unverified"
    assert report["decisions"]["status"] == "unverified"
    assert "empty_role" in codes(report)


def test_absent_role_is_unverified():
    report = audit_population(
        separated_manifest(), {"confirmation": ["s1"]}, overlap_rule=rule(["road", "scene"])
    )
    pair = report["decisions"]["pairs"][0]
    assert pair["decision"] == "unverified"
    assert pair["dimensions"] == {"road": "unverified", "scene": "unverified"}
    assert report["decisions"]["status"] == "unverified"


def test_conflicting_scene_named_in_a_role_has_its_own_code():
    first = timed("s1", "road-1", 0, 10)
    changed = {**first, "road_id": "road-9"}
    report = audit_population(
        wrap([first, changed, timed("s2", "road-2", 50, 60)]),
        {"pilot": ["s1"], "confirmation": ["s2"]},
        overlap_rule=rule(["road"]),
    )
    assert "role_scene_conflicting" in codes(report)
    assert "role_scene_unknown" not in codes(report)
    assert report["decisions"]["pairs"][0]["decision"] == "unverified"


def test_all_roles_rejected_blocks_role_membership():
    report = audit_population(separated_manifest(), {"pilot": "s1", "training": ["s2"]})
    assert report["role_membership"]["status"] == "blocked"
    assert report["role_membership"]["code"] == "role_membership_rejected"


def test_audit_errors_never_leave_decisions_passed():
    data = wrap(
        [
            timed("s1", "road-1", 0, 10, stratum="urban"),
            timed("s2", "road-2", 50, 60),
            timed("s3", "road-1", 70, 80, stratum="highway"),
        ]
    )
    report = audit_population(
        data, {"pilot": ["s1"], "confirmation": ["s2"]}, overlap_rule=rule(["scene"])
    )
    assert "road_stratum_conflict" in codes(report)
    assert report["status"] == "blocked"
    assert report["decisions"]["pairs"][0]["decision"] == "separated"
    assert report["decisions"]["status"] == "blocked"
    assert report["decisions"]["code"] == "audit_blocked"


def test_road_without_evidence_is_unverified_for_separation():
    data = wrap(
        [
            timed("s1", "road-1", 0, 10, road_evidence_ref=None),
            timed("s2", "road-2", 50, 60),
        ]
    )
    report = audit_population(
        data, {"pilot": ["s1"], "confirmation": ["s2"]}, overlap_rule=rule(["road"])
    )
    facts = report["role_membership"]["intersections"][0]["dimensions"]
    assert facts["road"]["status"] == "unverified"
    assert report["decisions"]["status"] == "unverified"
    assert report["role_membership"]["roles"]["pilot"]["unverified"]["road"] == 1


def test_shared_unevidenced_road_is_still_overlapping():
    data = wrap(
        [
            timed("s1", "road-1", 0, 10, road_evidence_ref=None),
            timed("s2", "road-1", 50, 60),
        ]
    )
    report = audit_population(
        data, {"pilot": ["s1"], "confirmation": ["s2"]}, overlap_rule=rule(["road"])
    )
    assert report["decisions"]["code"] == "overlap_rule_violated"


def test_huge_integer_interval_bound_is_invalid_not_overflow():
    report = audit_population(wrap([timed("s1", "road-1", 0, 10**400)]))
    assert "invalid_scene_record" in codes(report)
    assert report["status"] == "blocked"
