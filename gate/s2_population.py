"""S2 population and contamination audit (readiness CPU-03; build-spec A5).

Serves S2.1/S2.5/S2.6 [C] (G1, G2, G3·R1). Agent-authored evaluator engineering
under D-018/D-024.

``audit_population`` describes a declared scene population: counts by road, scene,
stratum and split label; missing identities; conflicting road/stratum assignments;
changed content under one scene ID; duplicate content under different IDs; clip and
half-open temporal overlap within a capture; and road/clip/temporal/scene overlap
between the roles pilot, instrument, development and confirmation. Unknown lineage
is ``unverified``, never clean. The original ``split_label`` is preserved verbatim;
the presence of ``alpamayo_aligned`` is a label, never a contamination finding.

Verification level per function (build-spec section 1):

- ``half_open_overlap`` - L1: literal interval known answers.
- ``audit_population`` - L1 over synthetic manifests. An audit of actual licensed
  catalog metadata is data-gated (L3: ``qualify_on_host``/after licensed access).

Interface reconciliation (recorded, not silently chosen): build-spec A5 names
``audit_population(manifest, roles=None)``; readiness CPU-03 names
``audit_population(catalog, assignments, role_membership, overlap_rule)``. The
manifest carries the catalog and scene assignments, ``roles`` is the role
membership, and the owner rule is the keyword ``overlap_rule``.

Owner inputs (no defaults, never chosen here):

- ``overlap_rule`` - Huy's overlap/exclusion rule. ``None`` sets
  ``decisions`` to ``blocked: overlap_rule_unregistered`` while every fact is still
  computed. Schema (exact keys): ``schema_version`` (1), ``rule_id``, ``source_ref``,
  ``separated_roles`` (nonempty list of distinct two-role pairs) and ``dimensions``
  (nonempty distinct subset of ``DIMENSIONS``). Any shared fact in a named dimension
  is ``violated``; otherwise any unknown lineage is ``unverified``; otherwise
  ``separated``. The rule hash proves bytes, not that Huy approved it.
- roads, strata, roles and scene lists are the caller's registered records.

Manifest schema (``artifact_kind`` ``s2_population_manifest``, version 1, exact
keys ``artifact_kind, schema_version, scope, catalog_revision, scenes``). Each scene
has exactly ``scene_id``, ``asset_sha256`` (64 hex or null until authorized access),
``road_id``, ``road_evidence_ref``, ``stratum``, ``split_label`` and ``source``
(``capture_id``, ``clip_id``, ``interval``). ``interval`` is null or exactly
``start``, ``end`` (finite numbers, ``start < end``), ``unit`` and ``convention``.
Only ``half_open`` is implemented; another convention, or different units within one
capture, leaves temporal overlap ``unverified`` (no unit conversion is invented).

Agent technical defaults (D-018): temporal overlap is compared only within one
``capture_id``; a scene listed in both roles overlaps itself temporally over its
whole interval (reported as the self-pair ``[id, id]``), or is ``unverified`` when
its interval is unknown; a road ID without ``road_evidence_ref`` is unknown lineage
for road separation (a shared road ID is still overlap); a role with a dropped
(unknown, conflicting) member or no members is incomplete, so its rule pairs can be
``violated`` or ``unverified`` but never ``separated``; any error-severity issue
blocks the decisions (``audit_blocked``) unless a violation is already reported, and
any rejected role or member sets ``role_membership`` to ``blocked``; identical
repeated scene rows (for example a replicate-expanded listing) are idempotent with a
warning; repeated role entries are idempotent; duplicate asset bytes under two scene
IDs reject (they would be counted as two scenes). ``manifest`` and ``roles`` hashes
are over their canonical sorted form, so input order does not change the report; the
rule hash is over the supplied value.

This module never clusters roads, replaces scenes, merges strata, claims road
independence or unseen-training-data cleanliness, reads licensed data, performs
network or subprocess calls, or imports protected inference.
"""

import hashlib
import math
import re
from copy import deepcopy
from itertools import combinations
from pathlib import Path

from gate.s2_identity import canonical_json_bytes, identity_hash

ARTIFACT_KIND = "s2_population_audit"
MANIFEST_KIND = "s2_population_manifest"
SCHEMA_VERSION = 1
SCOPES = ("synthetic", "public_source_fixture", "actual")
ROLES = ("confirmation", "development", "instrument", "pilot")
DIMENSIONS = ("clip", "road", "scene", "temporal")

_MANIFEST_KEYS = {"artifact_kind", "schema_version", "scope", "catalog_revision", "scenes"}
_SCENE_KEYS = {
    "scene_id",
    "asset_sha256",
    "road_id",
    "road_evidence_ref",
    "stratum",
    "split_label",
    "source",
}
_SOURCE_KEYS = {"capture_id", "clip_id", "interval"}
_INTERVAL_KEYS = {"start", "end", "unit", "convention"}
_RULE_KEYS = {"schema_version", "rule_id", "source_ref", "separated_roles", "dimensions"}
_OPTIONAL_TEXT = ("road_id", "road_evidence_ref", "stratum", "split_label")


def _source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _text(value):
    return type(value) is str and bool(value.strip())


def _optional_text(value):
    return value is None or _text(value)


def _number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int beyond float range is not a supported bound
        return False


def half_open_overlap(start_a, end_a, start_b, end_b):
    """Return the length of ``[start_a, end_a) ∩ [start_b, end_b)`` (L1).

    ``[0,10) ∩ [8,12)`` is 2; ``[0,10) ∩ [10,12)`` is 0 (touching is not overlap).
    """
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def _scene_problem(row):
    if type(row) is not dict or set(row) != _SCENE_KEYS:
        return "Scene keys differ from the manifest schema"
    if not _text(row["scene_id"]):
        return "scene_id must be text"
    digest = row["asset_sha256"]
    if digest is not None and (
        type(digest) is not str or not re.fullmatch(r"[0-9a-f]{64}", digest)
    ):
        return "asset_sha256 must be 64 lowercase hex characters or null"
    if not all(_optional_text(row[key]) for key in _OPTIONAL_TEXT):
        return "road_id, road_evidence_ref, stratum and split_label must be text or null"
    source = row["source"]
    if type(source) is not dict or set(source) != _SOURCE_KEYS:
        return "source requires exactly capture_id, clip_id and interval"
    if not (_optional_text(source["capture_id"]) and _optional_text(source["clip_id"])):
        return "capture_id and clip_id must be text or null"
    interval = source["interval"]
    if interval is not None:
        if type(interval) is not dict or set(interval) != _INTERVAL_KEYS:
            return "interval requires exactly start, end, unit and convention"
        if not (_number(interval["start"]) and _number(interval["end"])):
            return "interval bounds must be finite numbers"
        if not interval["start"] < interval["end"]:
            return "interval requires start < end"
        if not (_text(interval["unit"]) and _text(interval["convention"])):
            return "interval unit and convention must be text"
    return None


def _rule_problem(rule):
    if (
        type(rule) is not dict
        or set(rule) != _RULE_KEYS
        or type(rule["schema_version"]) is not int
        or rule["schema_version"] != 1
        or not _text(rule["rule_id"])
        or not _text(rule["source_ref"])
    ):
        return "Rule requires exactly schema_version 1, rule_id, source_ref, pairs, dimensions"
    dimensions = rule["dimensions"]
    if (
        type(dimensions) is not list
        or not dimensions
        or any(name not in DIMENSIONS for name in dimensions)
        or len(set(dimensions)) != len(dimensions)
    ):
        return f"dimensions must be a nonempty distinct subset of {DIMENSIONS}"
    pairs = rule["separated_roles"]
    if type(pairs) is not list or not pairs:
        return "separated_roles must be a nonempty list"
    seen = set()
    for pair in pairs:
        if (
            type(pair) is not list
            or len(pair) != 2
            or any(name not in ROLES for name in pair)
            or pair[0] == pair[1]
        ):
            return f"each separated pair names two distinct roles from {ROLES}"
        key = tuple(sorted(pair))
        if key in seen:
            return "separated_roles repeats a pair"
        seen.add(key)
    return None


def audit_population(manifest, roles=None, *, overlap_rule=None):
    """Return a deterministic CPU-03 population/contamination artifact.

    Structural envelope errors (wrong manifest kind/version/keys/scope, non-JSON
    values, ``roles`` not a mapping) raise ``ValueError``. Row-level defects become
    issues; ``status`` is ``blocked`` when any error-severity issue exists. Overlap
    *facts* are always computed; overlap *decisions* need ``overlap_rule``.
    """
    if type(manifest) is not dict or set(manifest) != _MANIFEST_KEYS:
        raise ValueError("manifest keys differ from the population manifest schema")
    if manifest["artifact_kind"] != MANIFEST_KIND:
        raise ValueError(f"manifest artifact_kind must be {MANIFEST_KIND}")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("unsupported population manifest schema_version")
    if manifest["scope"] not in SCOPES:
        raise ValueError(f"manifest scope must be one of {SCOPES}")
    if not _text(manifest["catalog_revision"]):
        raise ValueError("catalog_revision must be text")
    if type(manifest["scenes"]) is not list:
        raise ValueError("manifest scenes must be a list")
    if roles is not None and type(roles) is not dict:
        raise ValueError("roles must be a mapping of role name to scene IDs, or None")
    # Canonical encoding rejects NaN, cycles and other non-JSON values.
    canonical_rows = sorted(canonical_json_bytes(row) for row in manifest["scenes"])
    manifest_hash = identity_hash(
        {
            **{key: manifest[key] for key in _MANIFEST_KEYS - {"scenes"}},
            "scenes_sha256": hashlib.sha256(b"\n".join(canonical_rows)).hexdigest(),
        }
    )
    roles_hash = None
    if roles is not None:
        canonical_json_bytes(roles)
        roles_hash = identity_hash(
            {
                name: sorted(set(members))
                if type(members) is list and all(type(m) is str for m in members)
                else members
                for name, members in roles.items()
            }
        )
    rule_hash = None if overlap_rule is None else identity_hash(overlap_rule)

    issues = []

    def issue(code, path, message, severity="error", refs=()):
        issues.append(
            {
                "code": code,
                "path": path,
                "severity": severity,
                "evidence_refs": sorted(refs),
                "message": message,
            }
        )

    # --- scene records -------------------------------------------------------------
    by_id, invalid_rows, duplicate_rows = {}, 0, 0
    for row in sorted(manifest["scenes"], key=canonical_json_bytes):
        problem = _scene_problem(row)
        if problem is not None:
            invalid_rows += 1
            issue("invalid_scene_record", f"scene:sha256:{identity_hash(row)}", problem)
            continue
        by_id.setdefault(row["scene_id"], []).append(row)
    scenes, conflicting = {}, []
    for scene_id in sorted(by_id):
        rows = by_id[scene_id]
        distinct = {canonical_json_bytes(row): row for row in rows}
        if len(distinct) == 1:
            scenes[scene_id] = rows[0]
            if len(rows) > 1:
                duplicate_rows += len(rows) - 1
                issue(
                    "duplicate_scene_record",
                    f"scene:{scene_id}",
                    "Identical scene record repeated; counted once.",
                    "warning",
                )
            continue
        conflicting.append(scene_id)
        assets = {row["asset_sha256"] for row in distinct.values()}
        if len(assets) > 1:
            issue(
                "asset_changed_under_same_id",
                f"scene:{scene_id}",
                "One scene ID names different asset bytes.",
                refs=[digest for digest in assets if digest is not None],
            )
        else:
            issue("scene_record_conflict", f"scene:{scene_id}", "One scene ID, different records.")

    # --- counts and identities -----------------------------------------------------
    scenes_by_road, road_strata, scenes_by_stratum, labels = {}, {}, {}, {}
    missing = dict.fromkeys(
        (
            "asset_sha256",
            "road_id",
            "road_evidence_ref",
            "stratum",
            "split_label",
            "capture_id",
            "clip_id",
            "interval",
        ),
        0,
    )
    by_asset, unlabelled = {}, 0
    for scene_id, row in scenes.items():
        source = row["source"]
        for key in ("asset_sha256", *_OPTIONAL_TEXT):
            missing[key] += row[key] is None
        for key in _SOURCE_KEYS:
            missing[key] += source[key] is None
        road, stratum = row["road_id"], row["stratum"]
        if road is not None:
            scenes_by_road[road] = scenes_by_road.get(road, 0) + 1
            if stratum is not None:
                road_strata.setdefault(road, set()).add(stratum)
        if stratum is not None:
            scenes_by_stratum[stratum] = scenes_by_stratum.get(stratum, 0) + 1
        label = row["split_label"]
        if label is None:
            unlabelled += 1
        else:
            labels[label] = labels.get(label, 0) + 1
        if row["asset_sha256"] is not None:
            by_asset.setdefault(row["asset_sha256"], []).append(scene_id)
    road_conflicts = sorted(road for road, strata in road_strata.items() if len(strata) > 1)
    for road in road_conflicts:
        issue(
            "road_stratum_conflict",
            f"road:{road}",
            f"Road belongs to several strata: {sorted(road_strata[road])}.",
        )
    roads_by_stratum = {}
    for road, strata in road_strata.items():
        for stratum in strata:
            roads_by_stratum[stratum] = roads_by_stratum.get(stratum, 0) + 1
    duplicate_content = []
    for digest in sorted(by_asset):
        if len(by_asset[digest]) > 1:
            ids = sorted(by_asset[digest])
            duplicate_content.append({"asset_sha256": digest, "scene_ids": ids})
            issue(
                "duplicate_content_different_ids",
                f"asset:{digest}",
                "Identical asset bytes under several scene IDs.",
                refs=ids,
            )

    # --- lineage and temporal facts ------------------------------------------------
    temporal_known = {}
    temporal_unverified = []
    for scene_id, row in scenes.items():
        source = row["source"]
        interval = source["interval"]
        if source["capture_id"] is None or interval is None:
            temporal_unverified.append(scene_id)
            continue
        if interval["convention"] != "half_open":
            issue(
                "unsupported_interval_convention",
                f"scene:{scene_id}",
                "Only half_open intervals are implemented; temporal overlap unverified.",
                "warning",
            )
            temporal_unverified.append(scene_id)
            continue
        temporal_known[scene_id] = (source["capture_id"], interval)
    temporal_overlaps, unit_mismatches = [], []
    overlap_pairs, unverified_pairs = set(), set()
    by_capture = {}
    for scene_id in sorted(temporal_known):
        by_capture.setdefault(temporal_known[scene_id][0], []).append(scene_id)
    capture_pairs = sorted(
        pair for members in by_capture.values() for pair in combinations(members, 2)
    )
    for first, second in capture_pairs:
        capture_a, interval_a = temporal_known[first]
        interval_b = temporal_known[second][1]
        if interval_a["unit"] != interval_b["unit"]:
            unit_mismatches.append(
                {
                    "scene_ids": [first, second],
                    "capture_id": capture_a,
                    "reason": "interval_unit_mismatch",
                }
            )
            unverified_pairs.add((first, second))
            continue
        overlap = half_open_overlap(
            interval_a["start"], interval_a["end"], interval_b["start"], interval_b["end"]
        )
        if overlap > 0:
            overlap_pairs.add((first, second))
            temporal_overlaps.append(
                {
                    "scene_ids": [first, second],
                    "capture_id": capture_a,
                    "unit": interval_a["unit"],
                    "overlap": overlap,
                    "intervals": [
                        [interval_a["start"], interval_a["end"]],
                        [interval_b["start"], interval_b["end"]],
                    ],
                }
            )
    by_clip = {}
    for scene_id, row in scenes.items():
        clip = row["source"]["clip_id"]
        if clip is not None:
            by_clip.setdefault(clip, []).append(scene_id)
    shared_clips = [
        {"clip_id": clip, "scene_ids": sorted(ids)}
        for clip, ids in sorted(by_clip.items())
        if len(ids) > 1
    ]

    # --- role membership and intersections -----------------------------------------
    role_members, incomplete_roles, role_errors = {}, set(), 0
    role_section = {"status": "not_run", "code": "roles_not_supplied", "roles": {}}
    role_section["intersections"] = []
    if roles is not None:
        for name in sorted(roles, key=str):
            members = roles[name]
            if name not in ROLES:
                role_errors += 1
                issue("unknown_role", f"role:{name}", f"Role must be one of {ROLES}.")
                continue
            if type(members) is not list or not all(_text(member) for member in members):
                role_errors += 1
                issue("invalid_role_membership", f"role:{name}", "Members must be scene IDs.")
                continue
            known = set()
            for member in sorted(set(members)):
                if member not in scenes:
                    code = (
                        "role_scene_conflicting" if member in conflicting else "role_scene_unknown"
                    )
                    role_errors += 1
                    issue(code, f"role:{name}", f"Scene {member} is not an audited scene.")
                    continue
                known.add(member)
            if not members:
                issue(
                    "empty_role",
                    f"role:{name}",
                    "Role lists no scenes; its separation cannot be established.",
                    "warning",
                )
            # A dropped member or an empty list leaves the role's true membership unknown.
            if len(known) != len(set(members)) or not known:
                incomplete_roles.add(name)
            role_members[name] = known
        if role_errors:
            role_section.update(status="blocked", code="role_membership_rejected")
        else:
            role_section.update(status="passed", code=None)

    def road_unknown(scene_id):
        row = scenes[scene_id]
        return row["road_id"] is None or row["road_evidence_ref"] is None

    def dimension_facts(first_ids, second_ids):
        both = first_ids | second_ids
        shared_scenes = sorted(first_ids & second_ids)
        facts = {"scene": {"status": "overlapping" if shared_scenes else "disjoint"}}
        facts["scene"]["shared"] = shared_scenes
        for dimension, field in (("road", "road_id"), ("clip", "clip_id")):

            def values(ids, field=field):
                return {
                    scenes[i][field] if field == "road_id" else scenes[i]["source"][field]
                    for i in ids
                }

            left, right = values(first_ids), values(second_ids)
            shared = sorted((left & right) - {None})
            if dimension == "road":
                # A road ID without its evidence reference is unknown lineage for separation.
                unknown_lineage = any(road_unknown(i) for i in both)
            else:
                unknown_lineage = None in left | right
            status = "overlapping" if shared else ("unverified" if unknown_lineage else "disjoint")
            facts[dimension] = {"status": status, "shared": shared}
        pairs = {
            tuple(sorted((a, b)))
            for a in first_ids
            for b in second_ids
            if a != b and tuple(sorted((a, b))) in overlap_pairs
        }
        # A scene in both roles overlaps itself over its whole known interval; when its
        # interval is unknown it is already in temporal_unverified below.
        pairs |= {(i, i) for i in shared_scenes if i in temporal_known}
        pairs = sorted(pairs)
        unknown = any(i in temporal_unverified for i in both) or any(
            tuple(sorted((a, b))) in unverified_pairs
            for a in first_ids
            for b in second_ids
            if a != b
        )
        status = "overlapping" if pairs else ("unverified" if unknown else "disjoint")
        facts["temporal"] = {"status": status, "shared": [list(pair) for pair in pairs]}
        return facts

    intersections = {}
    for first, second in combinations(sorted(role_members), 2):
        facts = dimension_facts(role_members[first], role_members[second])
        intersections[(first, second)] = facts
        role_section["intersections"].append({"roles": [first, second], "dimensions": facts})
    role_section["roles"] = {
        name: {
            "scene_ids": sorted(ids),
            "roads": sorted({scenes[i]["road_id"] for i in ids} - {None}),
            "membership_complete": name not in incomplete_roles,
            "unverified": {
                "road": sum(road_unknown(i) for i in ids),
                "clip": sum(scenes[i]["source"]["clip_id"] is None for i in ids),
                "temporal": sum(i in temporal_unverified for i in ids),
            },
        }
        for name, ids in sorted(role_members.items())
    }

    # --- owner rule decisions ------------------------------------------------------
    decisions = {"status": "blocked", "code": "overlap_rule_unregistered", "rule": None}
    decisions["pairs"] = []
    if overlap_rule is not None:
        decisions["rule"] = {
            "rule_id": overlap_rule.get("rule_id") if type(overlap_rule) is dict else None,
            "sha256": rule_hash,
            "status": "supplied_not_authenticated",
        }
        problem = _rule_problem(overlap_rule)
        if problem is not None:
            issue("invalid_overlap_rule", "overlap_rule", problem)
            decisions["code"] = "invalid_overlap_rule"
        elif roles is None:
            decisions["code"] = "roles_not_supplied"
        else:
            verdicts = {"overlapping": "violated", "unverified": "unverified"}
            for pair in sorted(tuple(sorted(p)) for p in overlap_rule["separated_roles"]):
                incomplete = sorted(name for name in pair if name in incomplete_roles)
                if any(name not in role_members for name in pair):
                    outcome = dict.fromkeys(sorted(overlap_rule["dimensions"]), "unverified")
                else:
                    facts = intersections[pair]
                    outcome = {
                        name: verdicts.get(facts[name]["status"], "separated")
                        for name in sorted(overlap_rule["dimensions"])
                    }
                    if incomplete:
                        # Known shared facts stay violations; separation is not established.
                        outcome = {
                            name: "unverified" if value == "separated" else value
                            for name, value in outcome.items()
                        }
                worst = (
                    "violated"
                    if "violated" in outcome.values()
                    else "unverified"
                    if "unverified" in outcome.values()
                    else "separated"
                )
                decisions["pairs"].append(
                    {
                        "roles": list(pair),
                        "decision": worst,
                        "dimensions": outcome,
                        "incomplete_roles": incomplete,
                    }
                )
            results = {row["decision"] for row in decisions["pairs"]}
            audit_errors = any(item["severity"] == "error" for item in issues)
            if "violated" in results:
                decisions.update(status="blocked", code="overlap_rule_violated")
            elif audit_errors:
                decisions.update(status="blocked", code="audit_blocked")
            elif "unverified" in results:
                decisions.update(status="unverified", code="overlap_unverified")
            else:
                decisions.update(status="passed", code=None)

    scene_rows = [
        {
            "scene_id": scene_id,
            "asset_sha256": row["asset_sha256"],
            "road_id": row["road_id"],
            "road_evidence_ref": row["road_evidence_ref"],
            "stratum": row["stratum"],
            "split_label": row["split_label"],
            "source": deepcopy(row["source"]),
            "roles": sorted(name for name, ids in role_members.items() if scene_id in ids),
        }
        for scene_id, row in sorted(scenes.items())
    ]
    issues.sort(key=lambda item: (item["path"], item["code"], item["message"]))
    errors = any(item["severity"] == "error" for item in issues)
    return {
        "artifact_kind": ARTIFACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": manifest["scope"],
        "catalog_revision": manifest["catalog_revision"],
        "producer": {"module": "gate.s2_population", "source_sha256": _source_sha256()},
        "input_hashes": {"manifest": manifest_hash, "roles": roles_hash, "overlap_rule": rule_hash},
        "units_of_measure": {
            "counts": "distinct scene IDs or road IDs as named",
            "temporal_overlap": "declared interval unit of the compared scenes",
        },
        "status": "blocked" if errors else "passed",
        "status_meaning": "audit consistency only; never a contamination clearance",
        "counts": {
            "scenes": len(scenes),
            "roads": len(scenes_by_road),
            "strata": len(scenes_by_stratum),
            "scenes_by_road": dict(sorted(scenes_by_road.items())),
            "scenes_by_stratum": dict(sorted(scenes_by_stratum.items())),
            "roads_by_stratum": dict(sorted(roads_by_stratum.items())),
            "split_label_counts": dict(sorted(labels.items())),
            "unlabelled_scenes": unlabelled,
            "road_stratum_conflicts": road_conflicts,
            "conflicting_scene_ids": conflicting,
            "duplicate_content": duplicate_content,
            "invalid_scene_records": invalid_rows,
            "duplicate_scene_records": duplicate_rows,
        },
        "lineage": {
            "missing": missing,
            "temporal_unverified_scene_ids": sorted(temporal_unverified),
        },
        "scenes": scene_rows,
        "shared_clips": shared_clips,
        "temporal_overlaps": temporal_overlaps,
        "temporal_unverified_pairs": unit_mismatches,
        "role_membership": role_section,
        "decisions": decisions,
        "claims": {
            "road_independence": "not_established",
            "unseen_training_data_cleanliness": "not_established",
            "split_label_semantics": "label preserved; presence is not contamination",
        },
        "issues": issues,
    }
