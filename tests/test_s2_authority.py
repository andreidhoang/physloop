"""Known answers and negative controls for gate.s2_authority (hash-bound decision authority).

Fixture decision logs in ``tests/fixtures/s2_authority/`` are literal synthetic files; every
hash in them is the SHA-256 of a literal ASCII label (shown next to each constant), not of a real
artifact. The real ``docs/DECISIONS.md`` is read only to show it authorizes nothing today. No
test runs a process or touches the network.
"""

import hashlib
import re
from pathlib import Path

import pytest

from gate.s2_authority import (
    AUTHENTICITY,
    CLOSED_AUTHORITIES,
    SCHEMA_VERSION,
    parse_decision_entries,
    verify_authority,
    verify_records,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "s2_authority"
REAL_LOG = ROOT / "docs" / "DECISIONS.md"
PILOT = "b59b238b529029cc6841b05ce444cab0b99b69510438d37e619f5c599f083ee0"  # synthetic-pilot-packet
READINESS = "02f0725785c709d910339e5fb09c4aaa8a940cd37aef27636315611e3f7a1137"  # -readiness-report
OTHER = "5d687c931fe121319227f886f20d92f459e8d373a3251ee5e83abc9ed710e0d3"  # synthetic-other-packet
QUOTED = "cee7932aad8c569b76cd7c8bcd7d79c55ea984e042cc95fc8d74cbaa27fe96d2"  # -quoted-packet
FENCED = "874b585a236f3ca5a97c99d5b38e30f1e06644217ff821405233e1c29b266803"  # -fenced-packet
D019 = "d6c0423d3c60818ba9bc6c8fe410cd2efa92dbf7f6bc51a666e19944ce8202fc"  # synthetic-d019-packet
ZERO = "0" * 64
READY = (("s2_readiness_report", READINESS),)
# Hand-copied text of entry D-025 in decisions_base.md (lines 28-35), the entry_sha256 oracle.
D025_TEXT = (
    "- **D-025 · 2026-11-09 · Synthetic pilot rental approval.** Huy approves the bound packet.\n"
    f"  - authorizes: s2_pilot_packet sha256:{PILOT}\n"
    f"  - authorizes: s2_readiness_report sha256:{READINESS}\n"
    "  - engineering_only_diagnostic: true\n"
    "  - records: terms:harmonizer\n"
    "  - records: acquire:teacher\n"
    "  - **D-025 implementation addendum · 2026-11-09.** A nested addendum is part of D-025, "
    "not a new entry.\n"
    "    - records: host_setup\n"
)
D025_SHA = hashlib.sha256(D025_TEXT.encode("utf-8")).hexdigest()
ENTRY = "- **D-030 · 2026-11-12 · Synthetic.** Inline fixture entry.\n"


def fixture(name):
    return (FIXTURES / name).read_bytes()


def codes(result):
    return [issue["code"] for issue in result["issues"]]


def check(data, decision_id="D-025", kind="s2_pilot_packet", sha=PILOT, also=()):
    return verify_authority(data, decision_id=decision_id, artifact_kind=kind,
                            artifact_sha256=sha, also_required=also)


def inline(*body):
    """ENTRY plus body lines; an unindented body line is indented two columns, the entry's
    content column, so it is a continuation line of D-030 (strict grammar)."""
    return (ENTRY + "".join(("  " + line if line[:1] not in ("", " ", "\t") else line) + "\n"
                            for line in body)).encode("utf-8")


def test_fixture_hash_labels_are_literal_sha256_of_their_labels():
    for label, value in (("synthetic-pilot-packet", PILOT), ("synthetic-readiness-report",
                         READINESS), ("synthetic-other-packet", OTHER),
                         ("synthetic-quoted-packet", QUOTED), ("synthetic-fenced-packet", FENCED),
                         ("synthetic-d019-packet", D019)):
        assert hashlib.sha256(label.encode()).hexdigest() == value


# --- correct hash -----------------------------------------------------------------------------


def test_correct_hash_with_required_readiness_report_is_authorized():
    result = check(fixture("decisions_base.md"), also=READY)
    assert result["authorized"] is True
    assert result["artifact_kind"] == "s2_authority_check"
    assert result["schema_version"] == SCHEMA_VERSION == 1
    assert result["authenticity"] == AUTHENTICITY == "supplied_not_authenticated"
    assert result["decision_id"] == "D-025"
    assert result["closed_by"] is None
    assert result["issues"] == []
    assert result["entry_sha256"] == D025_SHA
    assert result["matched_lines"] == [
        {"line": 29, "text": f"- authorizes: s2_pilot_packet sha256:{PILOT}"},
        {"line": 30, "text": f"- authorizes: s2_readiness_report sha256:{READINESS}"},
    ]
    assert result["engineering_only_diagnostic"] is True
    assert result["decisions_sha256"] == hashlib.sha256(fixture("decisions_base.md")).hexdigest()


def test_correct_hash_without_also_required_is_authorized():
    result = check(fixture("decisions_base.md"))
    assert result["authorized"] is True
    assert [m["line"] for m in result["matched_lines"]] == [29]


def test_result_is_deterministic_and_input_is_not_mutated():
    data = bytearray(fixture("decisions_base.md"))
    before = bytes(data)
    assert check(data, also=READY) == check(before, also=READY)
    assert bytes(data) == before


# --- wrong hash, missing required artifact, other entry ---------------------------------------


def test_wrong_hash_is_refused_and_names_the_bound_hash():
    result = check(fixture("decisions_base.md"), sha=OTHER)
    assert result["authorized"] is False
    assert codes(result) == ["artifact_not_authorized", "authorized_by_other_entry"]
    assert PILOT in result["issues"][0]["message"]
    assert result["matched_lines"] == []


def test_wrong_kind_with_right_hash_is_refused():
    result = check(fixture("decisions_base.md"), kind="s2_full_sweep_packet")
    assert result["authorized"] is False
    assert codes(result) == ["artifact_not_authorized"]


def test_missing_also_required_line_is_refused():
    other_report = (("s2_readiness_report", OTHER),)
    result = check(fixture("decisions_base.md"), also=other_report)
    assert result["authorized"] is False
    assert codes(result) == ["artifact_not_authorized"]
    assert [m["line"] for m in result["matched_lines"]] == [29]
    absent_kind = check(fixture("decisions_base.md"), also=(("s4_launch_record", PILOT),))
    assert absent_kind["authorized"] is False


def test_hash_in_a_different_entry_is_refused():
    result = check(fixture("decisions_base.md"), sha=OTHER, decision_id="D-024")
    assert result["authorized"] is False
    assert codes(result) == ["no_authorizes_line", "authorized_by_other_entry"]
    assert "D-026" in result["issues"][1]["message"]
    assert "P-031" in result["issues"][1]["message"]


def test_entry_with_the_hash_authorizes_only_itself():
    assert check(fixture("decisions_base.md"), decision_id="D-026", sha=OTHER)["authorized"]
    assert not check(fixture("decisions_base.md"), decision_id="D-026")["authorized"]


def test_proposal_grant_line_never_authorizes():
    result = check(fixture("decisions_base.md"), decision_id="P-031", sha=OTHER)
    assert result["authorized"] is False
    assert codes(result) == ["invalid_decision_id"]


# --- D-019 closed, D-024 no authorizes line, later closure ------------------------------------


def test_d019_is_closed_by_static_table_even_with_matching_hash_line():
    assert CLOSED_AUTHORITIES == {"D-019": "D-020"}
    result = check(fixture("decisions_base.md"), decision_id="D-019", sha=D019)
    assert result["authorized"] is False
    assert result["closed_by"] == "D-020"
    assert codes(result) == ["authority_closed"]
    assert [m["line"] for m in result["matched_lines"]] == [11]


def test_d024_has_no_authorizes_line_despite_hash_in_prose():
    result = check(fixture("decisions_base.md"), decision_id="D-024")
    assert result["authorized"] is False
    assert result["closed_by"] is None
    assert codes(result) == ["no_authorizes_line", "authorized_by_other_entry"]


def test_later_closes_authority_line_refuses():
    result = check(fixture("decisions_closed_later.md"), also=READY)
    assert result["authorized"] is False
    assert result["closed_by"] == "D-028"
    assert codes(result) == ["authority_closed"]
    assert result["entry_sha256"] == D025_SHA
    parsed = parse_decision_entries(fixture("decisions_closed_later.md"))
    assert parsed["entries"]["D-028"][0]["closes_authority"] == [{"target": "D-025", "line": 57}]


def test_closure_before_or_inside_the_entry_is_an_anomaly_and_refuses():
    early = ("- **D-029 · 2026-11-11 · Early.** x\n  - closes_authority: D-030\n"
             f"{ENTRY}  - authorizes: s2_pilot_packet sha256:{PILOT}\n").encode()
    result = check(early, decision_id="D-030")
    assert result["authorized"] is False
    assert result["closed_by"] == "D-029"
    assert codes(result) == ["authority_closed", "closure_ordering_anomaly"]
    own = inline(f"  - authorizes: s2_pilot_packet sha256:{PILOT}", "  closes_authority: D-030")
    result = check(own, decision_id="D-030")
    assert result["authorized"] is False
    assert codes(result) == ["authority_closed", "closure_ordering_anomaly"]


def test_closure_counts_in_a_proposal_a_fence_or_outside_any_entry():
    grant = f"  - authorizes: s2_pilot_packet sha256:{PILOT}"
    for tail, closer in (
        ("- **P-040 · proposed 2026-11-12 · Proposal.** x\n  closes_authority: D-030\n", "P-040"),
        ("## Later\n\ncloses_authority: D-030\n", "line 5"),
        ("- **D-031 · 2026-11-12 · Fence.** x\n  ```\n  closes_authority: D-030\n  ```\n",
         "D-031"),
    ):
        result = check(inline(grant) + tail.encode(), decision_id="D-030")
        assert result["authorized"] is False, tail
        assert result["closed_by"] == closer
    unrelated = inline(grant) + b"- **D-031 \xc2\xb7 2026-11-12 \xc2\xb7 x.** y\n" \
        b"  closes_authority: D-029\n"
    assert check(unrelated, decision_id="D-030")["authorized"] is True


@pytest.mark.parametrize("line", [
    "  - closes_authority: D-030.",
    "  - Closes_authority: D-030",
    "  - closes_authority: D-30",
    "  - closes_authority:D-030",
    "  - closes_authority: D-030, D-031",
])
def test_mistyped_later_closure_naming_the_decision_refuses(line):
    data = inline(f"  - authorizes: s2_pilot_packet sha256:{PILOT}") + \
        ("- **D-031 · 2026-11-12 · Closure typo.** x\n" + line + "\n").encode()
    result = check(data, decision_id="D-030")
    assert result["authorized"] is False, line
    assert result["closed_by"] == "D-031"
    assert "malformed_closure_line" in codes(result)
    other = check(data.replace(b"D-030 \xc2\xb7", b"D-032 \xc2\xb7"), decision_id="D-032")
    assert other["authorized"] is True, line
    assert codes(parse_decision_entries(data)) == ["malformed_closure_line"]


# --- duplicates, malformed headings, nested addenda ------------------------------------------


def test_duplicate_decision_ids_refuse():
    result = check(fixture("decisions_duplicate.md"), also=READY)
    assert result["authorized"] is False
    assert codes(result) == ["duplicate_decision_id"]
    assert "28, 54" in result["issues"][0]["message"]
    assert result["entry_sha256"] is None
    parsed = parse_decision_entries(fixture("decisions_duplicate.md"))
    assert len(parsed["entries"]["D-025"]) == 2
    assert "duplicate_decision_id" in codes(parsed)
    other = check(fixture("decisions_duplicate.md"), decision_id="D-026", sha=OTHER)
    assert other["authorized"] is True


def test_nested_bold_addendum_is_not_a_second_entry():
    parsed = parse_decision_entries(fixture("decisions_base.md"))
    assert sorted(parsed["entries"]) == ["D-001", "D-019", "D-020", "D-024", "D-025", "D-026",
                                         "D-027"]
    assert parsed["proposals"] == ["P-031"]
    assert parsed["issues"] == []
    assert [r["ref"] for r in parsed["entries"]["D-025"][0]["records"]] == [
        "terms:harmonizer", "acquire:teacher", "host_setup"]


def test_level3_heading_ends_the_entry():
    data = (ENTRY + f"### Addendum\n  - authorizes: s2_pilot_packet sha256:{PILOT}\n").encode()
    result = check(data, decision_id="D-030")
    assert result["authorized"] is False
    assert codes(result) == ["no_authorizes_line"]
    assert codes(parse_decision_entries(data)) == ["directive_outside_entry"]


@pytest.mark.parametrize("heading", [
    "- **D-030 2026-11-12 Missing separators.**",
    "- **D-030 · 2026-11-1 · Short date.**",
    "- **D-30 · 2026-11-12 · Two digits.**",
    "- **D-030 - 2026-11-12 - Hyphens.**",
])
def test_malformed_heading_of_the_requested_id_refuses(heading):
    data = (heading + f"\n  - authorizes: s2_pilot_packet sha256:{PILOT}\n").encode()
    result = check(data, decision_id="D-030")
    assert result["authorized"] is False
    assert "decision_not_found" in codes(result)


def test_malformed_heading_with_the_same_id_beside_a_valid_entry_refuses():
    data = inline(f"  - authorizes: s2_pilot_packet sha256:{PILOT}") + \
        b"- **D-030 2026-11-13 Typo copy.** x\n"
    result = check(data, decision_id="D-030")
    assert result["authorized"] is False
    assert codes(result) == ["malformed_decision_heading"]
    assert result["issues"][0]["severity"] == "error"


def test_indented_heading_is_not_an_entry():
    data = ("  " + ENTRY + f"  - authorizes: s2_pilot_packet sha256:{PILOT}\n").encode()
    assert codes(check(data, decision_id="D-030")) == ["decision_not_found"]


# --- exact line form: code spans, quotes, prose, fences, malformed grants --------------------


def test_hash_in_code_span_quote_prose_bold_or_fence_does_not_authorize():
    for sha in (QUOTED, FENCED):
        result = check(fixture("decisions_base.md"), decision_id="D-027", sha=sha)
        assert result["authorized"] is False
        assert "no_authorizes_line" in codes(result)
        assert result["matched_lines"] == []
    fenced = check(fixture("decisions_base.md"), decision_id="D-027", sha=FENCED)
    assert codes(fenced).count("directive_in_code_fence") == 2


@pytest.mark.parametrize("line", [
    f"> authorizes: s2_pilot_packet sha256:{PILOT}",
    f"`authorizes: s2_pilot_packet sha256:{PILOT}`",
    f"- `authorizes: s2_pilot_packet sha256:{PILOT}`",
    f"**authorizes:** s2_pilot_packet sha256:{PILOT}",
    f"- **Records:** authorizes: s2_pilot_packet sha256:{PILOT}",
    f"Approved: authorizes: s2_pilot_packet sha256:{PILOT}",
    f"# authorizes: s2_pilot_packet sha256:{PILOT}",
])
def test_non_line_forms_grant_nothing(line):
    result = check(inline(line), decision_id="D-030")
    assert result["authorized"] is False
    assert "no_authorizes_line" in codes(result)


@pytest.mark.parametrize("line", [
    f"authorizes: s2_pilot_packet sha256:{PILOT}.",
    f"authorizes: s2_pilot_packet sha256:{PILOT.upper()}",
    f"authorizes: s2_pilot_packet sha256:{PILOT[:63]}",
    f"authorizes: s2_pilot_packet sha256:{PILOT}0",
    f"authorizes: s2_pilot_packet {PILOT}",
    f"authorizes:  s2_pilot_packet sha256:{PILOT}",
    f"authorizes: S2_Pilot_Packet sha256:{PILOT}",
    f"authorizes: s2_pilot_packet sha256:{PILOT} and more",
    f"Authorizes: s2_pilot_packet sha256:{PILOT}",
    f"- - authorizes: s2_pilot_packet sha256:{PILOT}",
])
def test_malformed_grant_lines_refuse(line):
    good = f"authorizes: s2_pilot_packet sha256:{PILOT}"
    result = check(inline(good, line), decision_id="D-030")
    assert result["authorized"] is False, line
    assert "malformed_directive_line" in codes(result)


@pytest.mark.parametrize("line", [
    f"authorizes: s2_pilot_packet sha256:{PILOT}",
    f"  authorizes: s2_pilot_packet sha256:{PILOT}",
    f"\tauthorizes: s2_pilot_packet sha256:{PILOT}",
    f"- authorizes: s2_pilot_packet sha256:{PILOT}",
    f"  * authorizes: s2_pilot_packet sha256:{PILOT}",
    f"    + authorizes: s2_pilot_packet sha256:{PILOT}  ",
])
def test_exact_line_forms_authorize(line):
    result = check(inline(line), decision_id="D-030")
    assert result["authorized"] is True, line
    assert result["matched_lines"] == [{"line": 2, "text": line.strip()}]


def test_unterminated_fence_swallows_later_grant_lines():
    data = inline("  ```", f"  - authorizes: s2_pilot_packet sha256:{PILOT}")
    result = check(data, decision_id="D-030")
    assert result["authorized"] is False
    assert codes(result) == ["unterminated_code_fence", "directive_in_code_fence",
                             "no_authorizes_line"]


def test_two_hashes_for_one_kind_refuse():
    data = inline(f"- authorizes: s2_pilot_packet sha256:{PILOT}",
                  f"- authorizes: s2_pilot_packet sha256:{OTHER}")
    result = check(data, decision_id="D-030")
    assert result["authorized"] is False
    assert codes(result) == ["conflicting_authorizes_lines"]


def test_repeated_identical_grant_line_is_accepted():
    line = f"- authorizes: s2_pilot_packet sha256:{PILOT}"
    result = check(inline(line, line), decision_id="D-030")
    assert result["authorized"] is True
    assert [m["line"] for m in result["matched_lines"]] == [2, 3]


def test_engineering_only_flag_is_reported_not_interpreted():
    grant = f"- authorizes: s2_pilot_packet sha256:{PILOT}"
    assert check(inline(grant), decision_id="D-030")["engineering_only_diagnostic"] is False
    flagged = check(inline(grant, "engineering_only_diagnostic: true"), decision_id="D-030")
    assert flagged["authorized"] is True and flagged["engineering_only_diagnostic"] is True
    bad = check(inline(grant, "engineering_only_diagnostic: yes"), decision_id="D-030")
    assert bad["authorized"] is False and codes(bad) == ["malformed_directive_line"]


# --- line endings and malformed input --------------------------------------------------------


def test_crlf_fixture_verifies_like_lf():
    crlf = fixture("decisions_crlf.md")
    assert b"\r\n" in crlf
    assert crlf == fixture("decisions_base.md").replace(b"\n", b"\r\n")
    lf_result = check(fixture("decisions_base.md"), also=READY)
    crlf_result = check(crlf, also=READY)
    assert crlf_result["authorized"] is True
    assert crlf_result["entry_sha256"] == lf_result["entry_sha256"] == D025_SHA
    assert crlf_result["matched_lines"] == lf_result["matched_lines"]
    assert crlf_result["decisions_sha256"] != lf_result["decisions_sha256"]
    closed = fixture("decisions_closed_later.md").replace(b"\n", b"\r\n")
    assert check(closed)["closed_by"] == "D-028"


def test_lone_carriage_return_does_not_split_lines():
    data = ENTRY.encode() + f"note\rauthorizes: s2_pilot_packet sha256:{PILOT}\n".encode()
    assert check(data, decision_id="D-030")["authorized"] is False


def test_utf8_bom_is_ignored():
    data = b"\xef\xbb\xbf" + inline(f"authorizes: s2_pilot_packet sha256:{PILOT}")
    assert check(data, decision_id="D-030")["authorized"] is True


@pytest.mark.parametrize("data,code", [
    (b"\xff\xfe- **D-025", "decisions_not_utf8"),
    ("- **D-025 · 2026-11-09 · str input.**", "decisions_not_bytes"),
    (None, "decisions_not_bytes"),
    (12, "decisions_not_bytes"),
])
def test_malformed_decision_bytes_never_raise(data, code):
    result = check(data)
    assert result["authorized"] is False
    assert codes(result) == [code]
    assert parse_decision_entries(data)["issues"][0]["code"] == code
    assert verify_records(data, decision_id="D-025", refs=("host_setup",))["satisfied"] is False


def test_empty_file_finds_nothing():
    assert codes(check(b"")) == ["decision_not_found"]
    assert parse_decision_entries(b"")["entries"] == {}


@pytest.mark.parametrize("kwargs,code", [
    ({"decision_id": "D-19"}, "invalid_decision_id"),
    ({"decision_id": "d-025"}, "invalid_decision_id"),
    ({"decision_id": None}, "invalid_decision_id"),
    ({"decision_id": "D-0250"}, "invalid_decision_id"),
    ({"sha": PILOT.upper()}, "invalid_artifact_request"),
    ({"sha": "sha256:" + PILOT}, "invalid_artifact_request"),
    ({"sha": PILOT[:63]}, "invalid_artifact_request"),
    ({"sha": None}, "invalid_artifact_request"),
    ({"kind": "S2 pilot"}, "invalid_artifact_request"),
    ({"kind": ""}, "invalid_artifact_request"),
    ({"also": (("s2_readiness_report", READINESS.upper()),)}, "invalid_also_required"),
    ({"also": ("s2_readiness_report", READINESS)}, "invalid_also_required"),
    ({"also": "s2_readiness_report"}, "invalid_also_required"),
    ({"also": (("s2_readiness_report",),)}, "invalid_also_required"),
])
def test_invalid_requests_refuse_without_raising(kwargs, code):
    result = check(fixture("decisions_base.md"), **kwargs)
    assert result["authorized"] is False
    assert code in codes(result)


def test_issue_shape_follows_the_shared_contract():
    result = check(fixture("decisions_closed_later.md"))
    for issue in result["issues"]:
        assert set(issue) == {"code", "path", "severity", "evidence_refs", "message"}
        assert re.fullmatch(r"[a-z][a-z0-9_]*", issue["code"])


# --- verify_records --------------------------------------------------------------------------


def test_records_present_in_unclosed_entry_are_satisfied():
    result = verify_records(fixture("decisions_base.md"), decision_id="D-025",
                            refs=("terms:harmonizer", "acquire:teacher", "host_setup"))
    assert result["satisfied"] is True
    assert result["artifact_kind"] == "s2_records_check"
    assert result["authenticity"] == "supplied_not_authenticated"
    assert result["missing_refs"] == []
    assert result["entry_sha256"] == D025_SHA
    assert [m["line"] for m in result["matched_lines"]] == [32, 33, 35]


def test_missing_record_is_named():
    result = verify_records(fixture("decisions_base.md"), decision_id="D-025",
                            refs=("terms:harmonizer", "acquire:nurec_scenes"))
    assert result["satisfied"] is False
    assert result["missing_refs"] == ["acquire:nurec_scenes"]
    assert codes(result) == ["record_missing"]


def test_records_in_closed_entry_are_refused():
    d019 = verify_records(fixture("decisions_base.md"), decision_id="D-019",
                          refs=("terms:harmonizer",))
    assert d019["satisfied"] is False and d019["closed_by"] == "D-020"
    later = verify_records(fixture("decisions_closed_later.md"), decision_id="D-025",
                           refs=("terms:harmonizer",))
    assert later["satisfied"] is False and later["closed_by"] == "D-028"


def test_records_elsewhere_or_in_fence_do_not_count():
    other = verify_records(fixture("decisions_base.md"), decision_id="D-026",
                           refs=("terms:harmonizer",))
    assert other["satisfied"] is False and other["missing_refs"] == ["terms:harmonizer"]
    fenced = verify_records(fixture("decisions_base.md"), decision_id="D-027",
                            refs=("terms:renderer",))
    assert fenced["satisfied"] is False and fenced["missing_refs"] == ["terms:renderer"]


def test_records_refuse_on_duplicates_and_empty_or_invalid_refs():
    dup = verify_records(fixture("decisions_duplicate.md"), decision_id="D-025",
                         refs=("terms:harmonizer",))
    assert dup["satisfied"] is False and codes(dup) == ["duplicate_decision_id"]
    empty = verify_records(fixture("decisions_base.md"), decision_id="D-025", refs=())
    assert empty["satisfied"] is False and codes(empty) == ["no_refs"]
    for bad in (("Terms:harmonizer",), ("terms:",), ("terms harmonizer",), (None,), "host_setup"):
        result = verify_records(fixture("decisions_base.md"), decision_id="D-025", refs=bad)
        assert result["satisfied"] is False, bad


def test_malformed_records_line_refuses():
    data = inline("records: terms:harmonizer", "records: Terms:Harmonizer")
    result = verify_records(data, decision_id="D-030", refs=("terms:harmonizer",))
    assert result["satisfied"] is False
    assert codes(result) == ["malformed_directive_line"]


# --- the real decision log -------------------------------------------------------------------


GRANT_ANYWHERE = re.compile(r"authorizes: [a-z][a-z0-9_]* sha256:[0-9a-f]{64}")


def test_real_decisions_file_authorizes_nothing_today():
    real = (ROOT / "docs" / "DECISIONS.md").read_bytes()
    assert not GRANT_ANYWHERE.search(real.decode("utf-8"))
    assert b"records: " not in real and b"closes_authority" not in real
    parsed = parse_decision_entries(real)
    assert parsed["issues"] == []
    numbers = sorted(int(entry_id[2:]) for entry_id in parsed["entries"])
    assert numbers[0] == 1 and numbers == list(range(1, numbers[-1] + 1))
    assert numbers[-1] >= 24
    assert all(len(occurrences) == 1 for occurrences in parsed["entries"].values())
    assert all(o["authorizes"] == [] and o["records"] == []
               for occurrences in parsed["entries"].values() for o in occurrences)
    d019 = check(real, decision_id="D-019", sha=ZERO)
    assert d019["authorized"] is False and d019["closed_by"] == "D-020"
    d024 = check(real, decision_id="D-024", sha=ZERO)
    assert d024["authorized"] is False and d024["closed_by"] is None
    assert codes(d024) == ["no_authorizes_line"]
    assert check(real, decision_id="D-023", sha=ZERO)["closed_by"] is None
    for n in numbers:
        result = check(real, decision_id=f"D-{n:03d}", sha=ZERO)
        assert not result["authorized"]
        expected = ["authority_closed"] if n == 19 else []
        assert codes(result) == expected + ["no_authorizes_line"], n
    assert codes(check(real, decision_id="D-999", sha=ZERO)) == ["decision_not_found"]
    records = verify_records(real, decision_id="D-024", refs=("terms:harmonizer",))
    assert records["satisfied"] is False


# --- review regressions (F1-01..F1-06): fences, Unicode digits, closure typos, comments -------

FENCE = "`" * 3
GRANT = f"  - authorizes: s2_pilot_packet sha256:{PILOT}"


def severity_of(result, code):
    return [issue["severity"] for issue in result["issues"] if issue["code"] == code]


def test_top_level_fenced_draft_entry_never_authorizes():
    """A1c: a draft entry pasted inside a column-0 fence under a draft heading."""
    data = ("## Draft for Huy (not approved; paste into a new entry to approve)\n\n"
            f"{FENCE}\n"
            "- **D-025 · 2026-11-10 · Approve the S2 pilot rental.** Huy approves the record.\n"
            f"{GRANT}\n{FENCE}\n\n## Next section\n").encode()
    result = check(data)
    assert result["authorized"] is False
    assert result["matched_lines"] == []
    assert severity_of(result, "entry_inside_code_fence") == ["error"]
    parsed = parse_decision_entries(data)
    assert "entry_inside_code_fence" in codes(parsed)
    assert parsed["entries"]["D-025"][0]["authorizes"] == []


def test_column0_fence_keeps_later_column0_headings_as_fenced_content():
    """A1 / A1b: the heading line no longer resets the fence; grants after it stay fenced."""
    a1 = ("- **D-030 · 2026-11-12 · Real.** x\n" f"{FENCE}\n"
          "- **D-031 · 2026-11-12 · Draft for Huy, not approved.** y\n" f"{GRANT}\n{FENCE}\n")
    a1b = (a1.replace(f"{FENCE}\n- **D-031", f"Draft text to paste (NOT approved):\n{FENCE}\n"
                      "- **D-031") + f"Command to run after approval:\n{FENCE}\n"
           "python scripts/s2_lifecycle.py step --execute\n" f"{FENCE}\n")
    for doc in (a1, a1b):
        result = check(doc.encode(), decision_id="D-031")
        assert result["authorized"] is False, doc
        assert "entry_inside_code_fence" in codes(result)
        assert check(doc.encode(), decision_id="D-030")["authorized"] is False


def test_grant_after_a_fenced_draft_heading_does_not_attach_to_the_previous_entry():
    data = ("- **D-030 · 2026-11-12 · Real.** x\n" f"{FENCE}\n"
            "- **D-031 · 2026-11-12 · Draft.** y\n" f"{FENCE}\n{GRANT}\n").encode()
    assert check(data, decision_id="D-030")["authorized"] is False
    draft = check(data, decision_id="D-031")
    assert draft["authorized"] is False
    assert severity_of(draft, "entry_inside_code_fence") == ["error"]


def test_info_string_line_does_not_close_a_fence():
    """A2b: '```python' inside a fence is content (CommonMark: closers carry no info string)."""
    data = inline(f"  {FENCE}", "  example of a nested snippet", f"  {FENCE}python",
                  f"  authorizes: s2_pilot_packet sha256:{PILOT}", f"  {FENCE}", f"  {FENCE}")
    result = check(data, decision_id="D-030")
    assert result["authorized"] is False
    assert result["matched_lines"] == []
    assert "directive_in_code_fence" in codes(result)


def test_closer_with_other_character_or_fewer_marks_does_not_close():
    for opener, inner in ((f"  {FENCE}", "  ~~~"), ("  ````", f"  {FENCE}")):
        data = inline(opener, inner, f"  authorizes: s2_pilot_packet sha256:{PILOT}",
                      opener)
        result = check(data, decision_id="D-030")
        assert result["authorized"] is False, opener
        assert "directive_in_code_fence" in codes(result)


def test_outdented_closer_of_a_list_item_fence_refuses_later_entries():
    """A column-0 '```' ends the list item and opens a new fence in CommonMark; it is not a
    closer of the indented fence, so the draft that follows must not become live."""
    data = ("- **D-030 · 2026-11-12 · Real.** x\n" f"  {FENCE}\n  foo\n{FENCE}\n"
            "- **D-031 · 2026-11-12 · Draft.** y\n" f"{GRANT}\n{FENCE}\n").encode()
    result = check(data, decision_id="D-031")
    assert result["authorized"] is False
    assert "irregular_code_fence" in codes(result)


def test_entry_after_an_unclosed_list_item_fence_refuses():
    data = ("- **D-030 · 2026-11-12 · Real.** x\n" f"  {FENCE}\n  foo\n"
            "- **D-031 · 2026-11-12 · Next.** y\n" f"{GRANT}\n").encode()
    result = check(data, decision_id="D-031")
    assert result["authorized"] is False
    assert severity_of(result, "entry_inside_code_fence") == ["error"]


def test_regular_fence_in_an_earlier_entry_does_not_block_a_later_one():
    data = inline(f"  {FENCE}", "  example", f"  {FENCE}") + \
        ("- **D-031 · 2026-11-12 · Next.** y\n" f"{GRANT}\n").encode()
    result = check(data, decision_id="D-031")
    assert result["authorized"] is True
    assert result["issues"] == []


@pytest.mark.parametrize("digits", ["٠١٩", "０１９", "٠٢٥", "０２５"])
def test_non_ascii_digit_identifiers_are_not_entries(digits):
    did = "D-" + digits
    data = f"- **{did} · 2026-11-01 · Homoglyph.** x\n{GRANT}\n".encode()
    result = check(data, decision_id=did)
    assert result["authorized"] is False
    assert "invalid_decision_id" in codes(result)
    parsed = parse_decision_entries(data)
    assert parsed["entries"] == {}
    assert "malformed_decision_heading" in codes(parsed)


def test_lookalike_heading_beside_the_real_entry_is_not_an_entry():
    data = (f"- **D-025 · 2026-11-09 · Real.** x\n  - authorizes: s2_pilot_packet sha256:{OTHER}\n"
            f"- **D-０２５ · 2026-11-10 · Lookalike.** y\n{GRANT}\n").encode()
    parsed = parse_decision_entries(data)
    assert sorted(parsed["entries"]) == ["D-025"]
    assert "malformed_decision_heading" in codes(parsed)
    assert check(data)["authorized"] is False
    assert check(data, sha=OTHER)["authorized"] is True


def test_non_ascii_digits_in_a_date_or_closure_are_not_exact_forms():
    heading = f"- **D-030 · ２０２６-11-12 · Fullwidth year.** x\n{GRANT}\n".encode()
    assert "decision_not_found" in codes(check(heading, decision_id="D-030"))
    closure = inline(GRANT) + "- **D-031 · 2026-11-13 · Close.** z\n  - closes_authority: D-０３０\n" \
        .encode()
    result = check(closure, decision_id="D-030")
    assert result["authorized"] is False


@pytest.mark.parametrize("line", [
    "  - closes_authority: D025",
    "  - closes_authority: D 025",
    "  - closes_authority: D–025",
    "  - closes_authority: D-O25",
])
def test_closure_line_naming_no_parseable_decision_refuses_everything(line):
    data = (f"- **D-025 · 2026-11-09 · Approval.** x\n{GRANT}\n"
            "- **D-028 · 2026-11-10 · Pilot allocation closed.** y\n" + line + "\n").encode()
    result = check(data)
    assert result["authorized"] is False, line
    assert severity_of(result, "unattributable_closure_line") == ["error"]


@pytest.mark.parametrize("line", [
    "  - closes authority: D-025",
    "  - Closes-authority: D-025",
    "  - closes_authorty: D-025",
])
def test_misspelled_closure_keyword_naming_the_decision_closes_it(line):
    data = (f"- **D-025 · 2026-11-09 · Approval.** x\n{GRANT}\n"
            "- **D-028 · 2026-11-10 · Pilot allocation closed.** y\n" + line + "\n").encode()
    result = check(data)
    assert result["authorized"] is False, line
    assert result["closed_by"] == "D-028"


def test_closure_inside_an_html_comment_still_closes():
    data = inline(GRANT) + b"## End\n<!--\ncloses_authority: D-030\n-->\n"
    result = check(data, decision_id="D-030")
    assert result["authorized"] is False
    assert result["closed_by"] == "line 5"


def test_released_memoryview_never_raises():
    view = memoryview(inline(GRANT))
    view.release()
    result = check(view, decision_id="D-030")
    assert result["authorized"] is False
    assert codes(result) == ["decisions_not_bytes"]
    assert result["decisions_sha256"] is None
    assert codes(parse_decision_entries(view)) == ["decisions_not_bytes"]
    records = verify_records(view, decision_id="D-030", refs=("host_setup",))
    assert records["satisfied"] is False and "decisions_not_bytes" in codes(records)


def test_fence_opener_with_comment_marker_in_its_info_string_is_a_fence():
    data = inline(f"  {FENCE}<!--", "  -->", GRANT, f"  {FENCE}")
    result = check(data, decision_id="D-030")
    assert result["authorized"] is False
    assert "directive_in_code_fence" in codes(result)


# --- G5: CommonMark HTML comments, list-marker fences, HTML blocks, indented code -------------
# Each document below is a literal reverifier repro (R2-R5, the side-effect probe) or a hand-
# derived variant; the comment beside it states where CommonMark 0.31 places the grant line.

BARE = f"  authorizes: s2_pilot_packet sha256:{PILOT}"
DEEP = f"    authorizes: s2_pilot_packet sha256:{PILOT}"
REAL = "- **D-030 · 2026-11-12 · Real.**"


def refused_as(data, code, decision_id="D-030"):
    """Refused with ``code``; a masked grant is also absent from ``matched_lines`` (an
    irregular-layout refusal may still list the line it refuses)."""
    result = check(data, decision_id=decision_id)
    assert result["authorized"] is False
    assert code in codes(result), codes(result)
    if code.startswith("directive_in_"):
        assert result["matched_lines"] == []
    return result


def test_r2_code_span_comment_marker_does_not_hide_the_following_fence():
    """R2: '<!--' in a code span is inline text; the fence on line 2 opens, so the grant on
    line 4 is code. The old scanner opened a comment, missed the fence and authorized."""
    data = (f"{REAL} Comments use `<!--` markers.\n"
            f"  {FENCE}\n  example: -->\n{BARE}\n  {FENCE}\n  {FENCE}\n").encode()
    result = refused_as(data, "directive_in_code_fence")
    assert codes(result) == ["unterminated_code_fence", "directive_in_code_fence",
                             "no_authorizes_line"]


def test_r3_mid_paragraph_comment_is_ended_by_a_fence_that_interrupts_the_paragraph():
    data = (f"{REAL} x <!-- note\n"
            f"  {FENCE}\n  -->\n{BARE}\n  {FENCE}\n  {FENCE}\n").encode()
    result = refused_as(data, "directive_in_code_fence")
    assert codes(result) == ["unterminated_code_fence", "directive_in_code_fence",
                             "no_authorizes_line"]


class _HostileBytes(bytes):
    def __bytes__(self):
        raise RuntimeError("hostile __bytes__")


def test_bytes_subclass_whose_conversion_raises_never_raises():
    data = _HostileBytes(inline(GRANT))
    result = check(data, decision_id="D-030")
    assert result["authorized"] is False
    assert codes(result) == ["decisions_not_bytes"]
    assert codes(parse_decision_entries(data)) == ["decisions_not_bytes"]
    records = verify_records(data, decision_id="D-030", refs=("host_setup",))
    assert records["satisfied"] is False and "decisions_not_bytes" in codes(records)


def test_grant_indented_as_an_indented_code_block_never_authorizes():
    """Six columns after a blank line is an indented code block inside the entry's list item."""
    data = (f"{REAL} x\n\n      authorizes: s2_pilot_packet sha256:{PILOT}\n").encode()
    result = refused_as(data, "directive_in_indented_code")
    assert codes(result) == ["directive_in_indented_code", "no_authorizes_line"]
    records = verify_records((f"{REAL} x\n\n      records: host_setup\n").encode(),
                             decision_id="D-030", refs=("host_setup",))
    assert records["satisfied"] is False
    assert "directive_in_indented_code" in codes(records)


def test_nested_bullet_content_column_keeps_four_space_directives_live():
    data = inline("  - **Addendum.** nested", "    - records: host_setup",
                  "    - authorizes: s2_pilot_packet sha256:" + PILOT)
    result = check(data, decision_id="D-030")
    assert result["authorized"] is True and result["issues"] == []
    assert verify_records(data, decision_id="D-030", refs=("host_setup",))["satisfied"]


def test_real_decisions_file_has_no_html_or_code_blocks_and_authorizes_nothing():
    real = (ROOT / "docs" / "DECISIONS.md").read_bytes()
    parsed = parse_decision_entries(real)
    assert parsed["issues"] == []
    for n in sorted(int(entry_id[2:]) for entry_id in parsed["entries"]):
        result = check(real, decision_id=f"D-{n:03d}", sha=PILOT)
        assert result["authorized"] is False
        assert result["matched_lines"] == []
        assert not {"directive_in_code_fence", "directive_after_raw_html",
                    "directive_in_inline_span", "directive_after_link_definition",
                    "directive_in_indented_code", "irregular_code_fence"} & set(codes(result))


# --- G5 review round 2: R6-R8, attribution A1-A4, type-1 start tags ----------------------------
# Literal reviewer repros. Where CommonMark 0.31.2 places each grant is stated beside it; the
# reviewer's offline renderers (pandoc commonmark/gfm, markdown-it) agreed on every case.

HEAD = "- **D-030 · 2026-11-12 · Real.** x\n"
NEEDLE = f"authorizes: s2_pilot_packet sha256:{PILOT}"


@pytest.mark.parametrize("doc", [
    # R6a-R6d: '2.'/'0)' cannot interrupt a paragraph (CommonMark 5.2 exception 1), so the
    # marker line is paragraph text, '     ```' opens a fence and the grant is code.
    HEAD + f"  2. {FENCE}\n     {FENCE}\n  - {NEEDLE}\n",
    HEAD + f"  0) {FENCE}\n     {FENCE}\n  - {NEEDLE}\n",
    HEAD + f"  2. ~~~\n     ~~~\n  - {NEEDLE}\n",
    HEAD + f"  2. {FENCE}\n     {FENCE}\n  {NEEDLE}\n",
])
def test_r6_ordered_marker_not_starting_at_one_after_paragraph_text_refuses(doc):
    result = refused_as(doc.encode(), "irregular_code_fence")
    assert severity_of(result, "irregular_code_fence")[0] == "error"


def test_r6f_top_level_desync_never_authorizes_a_hidden_draft_entry():
    """Prose, '2. ```' (paragraph text), '   ```' (a fence opener): D-031 is code."""
    doc = ("Draft below for Huy (not approved):\n"
           f"2. {FENCE}\n   {FENCE}\n"
           "- **D-031 · 2026-11-13 · Approve the S2 pilot rental.** Draft text.\n"
           f"  - {NEEDLE}\n")
    refused_as(doc.encode(), "irregular_code_fence", decision_id="D-031")


def test_r6f_draft_appended_to_the_real_decision_log_never_authorizes():
    real = (ROOT / "docs" / "DECISIONS.md").read_bytes()
    tail = ("Draft for Huy, not approved:\n"
            f"2. {FENCE}\n   {FENCE}\n"
            "- **D-025 · 2026-11-10 · Approve the S2 pilot rental.** Draft.\n"
            f"  - {NEEDLE}\n")
    result = check(real + b"\n" + tail.encode(), decision_id="D-025")
    assert result["authorized"] is False


@pytest.mark.parametrize("doc", [
    # R8a-R8d: a thematic break is never a list item (CommonMark 4.1), so the content column
    # stays 2 and a grant indented 6-7 columns is an indented code block.
    HEAD + "  * * *\n       " + NEEDLE + "\n",
    HEAD + "  - - -\n\n       " + NEEDLE + "\n",
    HEAD + "  - ---\n      " + NEEDLE + "\n",
    HEAD + "  * * *\n       - " + NEEDLE + "\n",
    HEAD + "  -\t---\n      " + NEEDLE + "\n",
    HEAD + "   * ***\n       " + NEEDLE + "\n",
])
def test_r8_thematic_break_opens_no_content_column(doc):
    refused_as(doc.encode(), "directive_in_indented_code")


def test_r8_control_nested_bullet_keeps_its_content_column():
    result = check((HEAD + "  * x\n    " + NEEDLE + "\n").encode(), decision_id="D-030")
    assert result["authorized"] is True
    assert result["issues"] == []


@pytest.mark.parametrize("rule", ["* * *", "---", "___", " - - -"])
def test_column0_thematic_break_ends_the_entry(rule):
    """'- **D-030**' / '* * *' / '  - grant': the break ends the entry's list, so the grant
    belongs to a new list after it, not to D-030."""
    result = check((HEAD + rule + "\n" + GRANT + "\n").encode(), decision_id="D-030")
    assert result["authorized"] is False
    assert codes(result) == ["no_authorizes_line"]


@pytest.mark.parametrize("heading", [
    "* **D-031 · 2026-11-13 · Draft, not approved.** y\n  - ",
    "+ **D-031 · 2026-11-13 · Draft, not approved.** y\n  - ",
    "1. **D-031 · 2026-11-13 · Draft, not approved.** y\n   - ",
    "-  **D-031 · 2026-11-13 · Draft, not approved.** y\n   - ",
    "-\t**D-031 · 2026-11-13 · Draft, not approved.** y\n    - ",
    " - **D-031 · 2026-11-13 · Draft, not approved.** y\n   - ",
])
def test_a1_any_top_level_item_headed_by_an_identifier_ends_the_previous_entry(heading):
    """A1-A4: every renderer shows a separate top-level item headed D-031 that owns the grant."""
    data = (HEAD + heading + NEEDLE + "\n").encode()
    result = check(data, decision_id="D-030")
    assert result["authorized"] is False
    assert "no_authorizes_line" in codes(result)
    assert "malformed_decision_heading" in codes(result)
    parsed = parse_decision_entries(data)
    assert sorted(parsed["entries"]) == ["D-030"]
    assert "malformed_decision_heading" in codes(parsed)
    refused_as(data, "malformed_decision_heading", decision_id="D-031")


def test_a1_draft_item_appended_to_the_real_log_does_not_make_d024_authorize():
    real = (ROOT / "docs" / "DECISIONS.md").read_bytes()
    tail = ("* **D-025 · 2026-11-10 · Draft rental approval, NOT approved.** Draft.\n"
            f"  - {NEEDLE}\n")
    assert check(real, decision_id="D-024")["authorized"] is False
    result = check(real + b"\n" + tail.encode(), decision_id="D-024")
    assert result["authorized"] is False
    assert "no_authorizes_line" in codes(result)


# --- G5R: strict grammar (agent technical default, D-018) ---------------------------------------
# Literal re-verifier repros (review/G5-reverify probe_g501_paragraph.py,
# probe_g502_nested_break.py, probe_real_append2.py) and the heading misattribution. Each fails on the CommonMark-emulating
# scanner (all three offline renderers show the grant as code or under another heading) and must
# refuse under the strict grammar. Non-regression guards preserved from the block model are
# labelled as such.

TAIL6 = f"  2. {FENCE}\n     {FENCE}\n  - {NEEDLE}\n"


def nothing_authorized(data, also_ids=()):
    """No entry of ``data`` (nor any id in ``also_ids``) authorizes PILOT; returns the ids."""
    ids = sorted(set(parse_decision_entries(data)["entries"]) | set(also_ids))
    for decision_id in ids:
        assert check(data, decision_id=decision_id)["authorized"] is False, decision_id
    return ids


@pytest.mark.parametrize("doc", [
    HEAD + "  *\n" + TAIL6,  # P1 empty '*' item
    HEAD + "  +\n" + TAIL6,  # P1b
    HEAD + "  1.\n" + TAIL6,  # P1c
    HEAD + "  2)\n" + TAIL6,  # P1d
    HEAD + f"      {FENCE}\n      {FENCE}\n" + TAIL6,  # P2 fence pair at 6 columns
    HEAD + "      <!-- c -->\n" + TAIL6,  # P3 one-line comment at 6 columns
    HEAD + "      <pre>x</pre>\n" + TAIL6,  # P3b
    HEAD + "      <?x?>\n" + TAIL6,  # P3c
    HEAD + "      * * *\n" + TAIL6,  # P4 spaced thematic break at 6 columns
    HEAD + f"  *\n  2. <!-- note\n     {FENCE}\n     -->\n  - {NEEDLE}\n",  # P5
    HEAD + TAIL6,  # C1, refused before (non-regression guard)
])
def test_g501_ordered_marker_after_paragraph_text_never_authorizes(doc):
    nothing_authorized(doc.encode(), ("D-030",))


@pytest.mark.parametrize("before", ["*\n", "    <!-- c -->\n", ""])
def test_g501_top_level_desync_never_authorizes_a_hidden_draft_entry(before):
    """P6, P6b and the original R6f: '2. ```' after prose hides the D-031 draft as code."""
    doc = ("Draft below for Huy (not approved):\n" + before + f"2. {FENCE}\n   {FENCE}\n"
           "- **D-031 · 2026-11-13 · Approve the S2 pilot rental.** Draft text.\n"
           f"  - {NEEDLE}\n")
    nothing_authorized(doc.encode(), ("D-031",))


@pytest.mark.parametrize("before", ["*\n", "    <!-- c -->\n", ""])
def test_g501_draft_appended_to_the_real_log_never_authorizes(before):
    """probe_real_append2 (b'), (b''), (d) against the real docs/DECISIONS.md bytes."""
    tail = ("Draft for Huy, not approved:\n" + before + f"2. {FENCE}\n   {FENCE}\n"
            "- **D-025 · 2026-11-10 · Approve the S2 pilot rental.** Draft.\n"
            f"  - {NEEDLE}\n")
    nothing_authorized(REAL_LOG.read_bytes() + b"\n" + tail.encode(), ("D-025",))


@pytest.mark.parametrize("doc", [
    HEAD + "  - * * *\n         " + NEEDLE + "\n",  # Q1
    HEAD + "  - * * *\n\n         " + NEEDLE + "\n",  # Q2
    HEAD + "  * - - -\n         - " + NEEDLE + "\n",  # Q3
    HEAD + "  + * * *\n          " + NEEDLE + "\n",  # Q4
    HEAD + "  - - * * *\n           " + NEEDLE + "\n",  # Q5
    HEAD + "  - * * * *\n            " + NEEDLE + "\n",  # Q6
])
def test_g502_spaced_thematic_break_as_bullet_content_never_authorizes(doc):
    result = check(doc.encode(), decision_id="D-030")
    assert result["authorized"] is False
    assert severity_of(result, "directive_in_indented_code") == ["error"]


def test_g502_nested_break_appended_inside_the_real_d024_never_authorizes():
    """probe_real_append2 (c): the real log ends inside D-024's list item."""
    data = REAL_LOG.read_bytes() + ("  - * * *\n" f"         {NEEDLE}\n").encode()
    assert check(data, decision_id="D-024")["authorized"] is False
    nothing_authorized(data)


MISATTRIBUTED = [
    "- __D-025 · 2026-11-10 · Draft rental approval, not approved.__ Draft.\n",
    "- D-025 · 2026-11-10 · Draft rental approval, not approved. Draft.\n",
    "* __D-031 · 2026-11-13 · Draft, not approved.__ y\n",
    "- *D-031* · 2026-11-13 · Draft, not approved. y\n",
    "- **Draft D-031 · 2026-11-13 · Rental, not approved.** y\n",
]


@pytest.mark.parametrize("separator", ["", "\n"])
@pytest.mark.parametrize("heading", MISATTRIBUTED)
def test_g5r_differently_emphasised_draft_heading_never_authorizes_the_previous_entry(
        heading, separator):
    """Every renderer shows a separate item headed by the draft identifier: its grant belongs
    to that draft, never to the real D-024 that precedes it."""
    real = REAL_LOG.read_bytes()
    data = real + (separator + heading + f"  - {NEEDLE}\n").encode()
    result = check(data, decision_id="D-024")
    assert result["authorized"] is False
    assert codes(result) == ["malformed_decision_heading", "no_authorizes_line"]
    assert severity_of(result, "malformed_decision_heading") == ["warning"]
    parsed = parse_decision_entries(data)
    assert parsed["entries"]["D-024"][0]["authorizes"] == []
    nothing_authorized(data, ("D-025", "D-031"))
    assert check(real, decision_id="D-024")["closed_by"] is None


def test_g5r_real_log_still_authorizes_nothing_and_d019_stays_closed_by_d020():
    real = REAL_LOG.read_bytes()
    assert parse_decision_entries(real)["issues"] == []
    assert nothing_authorized(real)[-1] >= "D-024"
    d019 = check(real, decision_id="D-019", sha=ZERO)
    assert d019["closed_by"] == "D-020" and codes(d019)[0] == "authority_closed"


def test_g5r_column0_directive_ends_the_entry_and_counts_for_nothing():
    for line in (NEEDLE, f"- {NEEDLE}", f"* {NEEDLE}"):
        data = (ENTRY + line + "\n").encode()
        result = check(data, decision_id="D-030")
        assert result["authorized"] is False, line
        assert codes(result) == ["no_authorizes_line"]
        assert "directive_outside_entry" in codes(parse_decision_entries(data))


def test_g5r_one_column_line_after_a_blank_line_ends_the_entry():
    """' x' after a blank line ends the top-level list; the grant after it is a new list."""
    data = (ENTRY + "\n x\n" + GRANT + "\n").encode()
    result = check(data, decision_id="D-030")
    assert result["authorized"] is False
    assert codes(result) == ["no_authorizes_line"]


@pytest.mark.parametrize("marker", ["<span hidden>", "<!-- -->", "</b>", "<!x>", "<details>"])
def test_g5r_raw_html_refuses_every_later_directive(marker):
    data = (f"{REAL} x {marker}\n{GRANT}\n").encode()
    result = check(data, decision_id="D-030")
    assert result["authorized"] is False, marker
    assert severity_of(result, "directive_after_raw_html") == ["error"]
    assert result["matched_lines"] == []


def test_g5r_directives_before_raw_html_in_the_entry_still_count():
    data = inline(GRANT, "  Note: <b>bold</b>")
    result = check(data, decision_id="D-030")
    assert result["authorized"] is True
    assert result["issues"] == []
    assert "raw_html_marker" in codes(parse_decision_entries(data))


def test_g5r_raw_html_in_an_earlier_entry_refuses_later_entries():
    """Non-regression guard preserved from the block model: '  <!--' in D-030 is emitted raw
    before D-030's '</li>', so a browser hides D-031 although D-031 holds no '<'."""
    data = (f"{REAL} x\n  <!-- draft\n- **D-031 · 2026-11-13 · Approval.** y\n{GRANT}\n"
            "- **D-032 · 2026-11-13 · Later.** -->\n").encode()
    result = check(data, decision_id="D-031")
    assert result["authorized"] is False
    assert severity_of(result, "directive_after_raw_html") == ["error"]


def test_g5r_column0_comment_around_a_draft_entry_refuses():
    data = ("<!--\n- **D-031 · 2026-11-12 · Draft.** y\n" f"{GRANT}\n-->\n").encode()
    result = check(data, decision_id="D-031")
    assert result["authorized"] is False
    assert "directive_after_raw_html" in codes(result)


@pytest.mark.parametrize("opener", [f"  - {FENCE}", f"  1. {FENCE}", "  * ~~~", f"  - - {FENCE}"])
def test_g5r_fence_after_a_list_marker_is_irregular_and_refuses(opener):
    width = len(opener) - 3
    closer = opener[-3:]
    data = (f"{REAL} x\n{opener}\n{' ' * width}example\n{' ' * width}{closer}\n{GRANT}\n"
            ).encode()
    result = check(data, decision_id="D-030")
    assert result["authorized"] is False, opener
    assert severity_of(result, "irregular_code_fence") == ["error"]


@pytest.mark.parametrize("body", [
    (f"      {FENCE}", "      x", f"      {FENCE}", GRANT),  # opener at six columns
    (f"  {FENCE}a`b", "  x", f"  {FENCE}", GRANT),  # backtick in the info string: not a fence
    (f"  {FENCE}", "  x", f"    {FENCE}", GRANT),  # closer at another indentation
    (f"\t{FENCE}", "\tx", f"\t{FENCE}", GRANT),  # tab before the opener
])
def test_g5r_irregular_fence_shapes_refuse(body):
    result = check(inline(*body), decision_id="D-030")
    assert result["authorized"] is False, body
    assert severity_of(result, "irregular_code_fence") == ["error"]


def test_g5r_regular_fence_with_a_closer_lookalike_inside_still_closes_at_its_own_closer():
    data = inline(f"  {FENCE}", "  x", f"  {FENCE} still content", f"  {FENCE}", GRANT)
    result = check(data, decision_id="D-030")
    assert result["authorized"] is True and result["issues"] == []


def test_g5r_directive_window_is_two_to_five_columns():
    for width, live in ((2, True), (4, True), (5, True), (6, False), (8, False)):
        data = (ENTRY + "  - **Addendum.** nested\n" + " " * width + NEEDLE + "\n").encode()
        assert check(data, decision_id="D-030")["authorized"] is live, width


# --- G5 repros under the strict grammar -----------------------------------------------------------
# The block-model tests these replace asserted CommonMark placements of each document. Every
# literal document is kept; each now refuses for the stricter reason stated. Documents the block
# model authorized (controls such as a closed '<pre>' block or a '1.' list fence) are false
# refusals by design (agent technical default, D-018).

RAW_PAIRS = [("<pre>", "</pre>"), ("<PRE class=x>", "</Pre>"), ("<script>", "</script>"),
             ("<style>", "</style>"), ("<textarea>", "</textarea>"), ("<?php", "?>"),
             ("<!DOCTYPE html", ">"), ("<![CDATA[", "]]>")]
HTML_DOCS = [
    inline("  <!-- draft for Huy, uncomment to approve:", GRANT, "  -->"),
    inline(f"  <!-- authorizes: s2_pilot_packet sha256:{PILOT} -->"),
    inline("  <!-- note -->", GRANT),
    inline("  <!-- open", GRANT),
    (f"{REAL} x\n\n  <pre>\n  {FENCE}\n  </pre>\n\n  {FENCE}\n{BARE}\n  {FENCE}\n  {FENCE}\n"
     ).encode(),
    *[inline(f"  {opener}", GRANT, f"  {closer}") for opener, closer in RAW_PAIRS],
    *[inline(f"  {opener}", f"  {closer}", GRANT) for opener, closer in RAW_PAIRS],
    inline("  <pre>example</pre>", GRANT),
    inline("  <pre>", GRANT),
    (f"{REAL} x\n\n  <div>\n  {FENCE}\n\n  {FENCE}\n{GRANT}\n  {FENCE}\n").encode(),
    (f"{REAL} x\n  <x-note>\n  {FENCE}\n\n  {FENCE}\n{GRANT}\n  {FENCE}\n").encode(),
    inline("  <div>", GRANT, "  </div>"),
    inline("  <div>", "  </div>", "", GRANT),
    (f"{REAL} x <!-- c\n{BARE}\n  -->\n").encode(),
    (f"{REAL} x <!-- note\n\n{GRANT}\n").encode(),
    (f"{REAL} x <!-- note\n  - next item\n{GRANT}\n").encode(),
    (f"{REAL} x <!-- c\n      {FENCE}\n      {FENCE}\n{BARE}\n  -->\n").encode(),
    (f"{REAL} x\n  <!--\n  {FENCE}\n{BARE}\n  -->\n").encode(),
    (f"{REAL} x\n  <!--\n  {FENCE}\n  -->\n{GRANT}\n").encode(),
    (f"{REAL} x\n      <!-- c\n  {FENCE}\n  -->\n{GRANT}\n  {FENCE}\n").encode(),
    (f"{REAL} x\n  - <!-- note\n  {FENCE}\n  -->\n{GRANT}\n  {FENCE}\n").encode(),
    inline("  <!-- a --> <!-- b", GRANT),
    inline("  <!--", "  a --> <!-- b", GRANT),
    inline("  <pre>", "  <!--", "  </pre>", GRANT),
    inline("  <div>", "  <!--", "", GRANT),
    inline("  <pre>x</pre> <!-- y", GRANT),
    inline("  <pre>", "  </textarea>", GRANT),
    inline("  <script></style>", GRANT),
    inline("  <pre>", "  <!-- x -->", "  </pre>", GRANT),
    inline("  <!-- a --> <!-- b -->", GRANT),
    inline("  <div>", "  <!-- c -->", "", GRANT),
    (f"{REAL} x\n\n  <a title=\"x>y\">\n  {FENCE}\n\n  {FENCE}\n{GRANT}\n  {FENCE}\n").encode(),
    (HEAD + f"  2. <!-- note\n     {FENCE}\n     -->\n  - {NEEDLE}\n").encode(),
    *[inline(f"  {opener}", f"  </{opener[1:].split()[0].lower()}>", GRANT)
      for opener in ("<style", "<script", "<textarea", "<pre", "<style media=x",
                     "<style title=\">\"", "<STYLE")],
    *[inline(f"  {opener}", f"  </{name}>", GRANT)
      for opener, name in (("<style>", "style"), ("<script type=\"text/plain\">", "script"),
                           ("<pre class=x>", "pre"), ("<textarea rows='2'>", "textarea"))],
]


@pytest.mark.parametrize("data", HTML_DOCS)
def test_g5_html_repros_refuse_under_the_raw_html_rule(data):
    result = check(data, decision_id="D-030")
    assert result["authorized"] is False
    assert result["matched_lines"] == []
    assert "raw_html_marker" in codes(parse_decision_entries(data))


@pytest.mark.parametrize("doc", [
    f"<!--\n- **D-031 · 2026-11-12 · Draft.** y\n-->\n{GRANT}\n",
    f"<pre>\n- **D-031 · 2026-11-12 · Draft.** y\n</pre>\n{GRANT}\n",
    # The side-effect probe: a code-span '<!--' in D-030 now refuses D-031 too (known false
    # refusal; it reverses the block model's G5 fix so a raw '<!--' never hides a later entry).
    ("- **D-030 · 2026-11-12 · Grammar.** The verifier masks `<!--` comments.\n"
     "- **D-031 · 2026-11-13 · Approval.** y\n"
     f"  - authorizes: s2_pilot_packet sha256:{PILOT}\n"),
])
def test_g5_entry_after_raw_html_never_authorizes(doc):
    data = doc.encode()
    result = check(data, decision_id="D-031")
    assert result["authorized"] is False
    assert result["matched_lines"] == []
    assert "raw_html_marker" in codes(parse_decision_entries(data))


@pytest.mark.parametrize("doc", [
    f"{REAL} x\n  - {FENCE}\n{DEEP}\n    {FENCE}\n    {FENCE}\n",  # R4
    *[f"{REAL} x\n{marker}{FENCE}\n{' ' * (len(marker))}{NEEDLE}\n{' ' * len(marker)}{FENCE}\n"
      for marker in ("  * ", "  + ", "  1. ", "  1) ", "- - ")],
    f"{REAL} x\n  - ~~~\n    {NEEDLE}\n    ~~~\n",
    f"{REAL} x\n  - {FENCE}\n    foo\n  {FENCE}\n{BARE}\n  {FENCE}\n",
    f"{REAL} x\n  -     {FENCE}\n      {NEEDLE}\n",
    HEAD + f"\n  2. {FENCE}\n     {FENCE}\n  - {NEEDLE}\n",  # C6a, a real list to CommonMark
    HEAD + f"  1. {FENCE}\n     {FENCE}\n  - {NEEDLE}\n",  # C6b
])
def test_g5_list_marker_fence_repros_are_irregular(doc):
    result = check(doc.encode(), decision_id="D-030")
    assert result["authorized"] is False
    assert "irregular_code_fence" in codes(result)



@pytest.mark.parametrize("body", [
    # A multi-line code span: the heading paragraph's backtick runs over the bare grant line.
    ("  Example: `x", BARE, "  y`"),
    # GFM strikethrough over a paragraph continuation line.
    ("  ~~withdrawn", BARE, "  ~~"),
])
def test_g5r_bare_directive_continuing_a_span_paragraph_refuses(body):
    result = check(inline(*body), decision_id="D-030")
    assert result["authorized"] is False
    assert severity_of(result, "directive_in_inline_span") == ["error"]


def test_g5r_bullet_directive_after_a_span_character_still_counts():
    """A bullet item starts a new block, so no inline span from the line above reaches it."""
    for body in (("  Uses `code` here.", GRANT), ("  Example: `x", GRANT, "  y`"),
                 ("  `a`", "", BARE)):
        result = check(inline(*body), decision_id="D-030")
        assert result["authorized"] is True, body
        assert result["issues"] == []


@pytest.mark.parametrize("body", [
    ("  [note]: /url '", BARE, "  '"),  # a multi-line link title is not rendered
    ("  [^1]: footnote", GRANT),  # a footnote definition renders elsewhere
])
def test_g5r_link_or_footnote_definition_refuses_every_later_directive(body):
    data = inline(*body)
    result = check(data, decision_id="D-030")
    assert result["authorized"] is False
    assert severity_of(result, "directive_after_link_definition") == ["error"]
    assert "link_definition_marker" in codes(parse_decision_entries(data))
