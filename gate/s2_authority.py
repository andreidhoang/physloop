"""Hash-bound authority verification over the bytes of ``docs/DECISIONS.md``.

Purpose. An executing path (rental lifecycle, unit runner, guest bootstrap, S4 launcher) may act
only under a recorded decision entry that names the exact artifact it is about to act on by its
canonical SHA-256. This module is the pure verifier behind that rule; phase 2 of the D-024 build
wires it into those callers. Tickets: completeness-audit "Safety" findings (``s2_lifecycle``
accepted any ``- **D-NNN ·`` heading, including closed D-019 and no-rental D-024;
``s2_run_units`` and ``host_launch`` accepted free-text authority); CPU-10 / runbook step 2
authority gate; build-spec section 2 item 6 ("any executing path requires an explicit
recorded-authority input and refuses without it"); G5 hardening and its re-verification G5R
(G5-01 ordered-marker desync, G5-02 thematic break as bullet content, heading misattribution).

Decision-file grammar (deliberately strict; fail closed). The brief fixes the entry/grant/closure
line forms. Everything else is an ``agent technical default (D-018)`` that replaces the earlier
CommonMark emulation: instead of predicting how a renderer places a line, the verifier accepts
only the one layout ``docs/DECISIONS.md`` uses (top-level ``- **D-NNN · date · title.**``
bullets, indented continuation lines, no HTML) and refuses wherever rendering could be ambiguous.
A visible grant written in any other layout is falsely refused; that is the intended trade.

- Lines are split on LF only; one trailing CR per line is removed, so CRLF and LF copies of a file
  verify identically. Columns count a tab as advancing to the next multiple of 4.
- An *entry* starts only at a column-0 line ``- **D-NNN · YYYY-MM-DD · ...`` (U+00B7 separators;
  ``N``, ``Y``, ``M`` and ``D`` are ASCII digits only, so Arabic-Indic or fullwidth lookalikes are
  never entries). A proposal block starts at ``- **P-NNN · ...``. Every other non-blank line
  indented less than two columns (the entry's content column) ends the current block: another
  list marker of any kind, an ordered marker, a heading, a thematic break, prose, a fence, and a
  plain or differently emphasised heading (``- __D-025``, ``- D-025``, ``* __D-031``,
  ``- *D-031*``, ``- **Draft D-031``). Such a line that names a decision near its start is
  reported as ``malformed_decision_heading`` (blocking for that identifier); it is never an
  entry, so a grant under it belongs to no entry. Nested items (``  - **D-023 addendum · ...**``)
  stay inside their entry.
- ``entry_sha256`` is SHA-256 over the entry's CR-stripped lines (heading to the line before
  the line that ends it) joined by LF with one trailing LF, after dropping trailing blank lines.
- A grant line is exactly ``authorizes: <kind> sha256:<64 lowercase hex>``, indented two to five
  columns inside an entry, optionally after one ``-``/``*``/``+`` bullet and one space, with only
  trailing spaces/tabs after. ``records: <ref>`` and ``engineering_only_diagnostic: true`` follow
  the same line form. Two to five columns can never be an indented code block under any list
  item (that needs four columns past a content column of at least two), so a directive indented
  six or more columns is refused (``directive_in_indented_code``, blocking) and a column-0 or
  column-1 directive is outside every entry. Prose, lines written with a ``>`` marker, inline
  code spans, bold labels and headings never grant. A line that starts like a directive but is
  not in exact form is ``malformed_directive_line`` and, inside the requested entry, refuses.
- Fences. A line whose text after indentation and any list markers starts with three or more
  backticks or tildes opens a code region at any indentation, inside or outside an entry; its
  directives never count (``directive_in_code_fence``). Only a line of the opener's character,
  at least as long, at exactly the opener's column, with nothing but spaces/tabs after it and no
  list marker closes it. The fence is irregular (``irregular_code_fence``) when the opener
  follows a list marker, is indented six or more columns inside an entry or four or more outside
  one, has a tab before its marker or a backtick in a backtick info string, when a closer stands
  at another indentation, or when a non-blank line is less indented than the open fence; an
  unclosed fence is ``unterminated_code_fence``. Either refuses every entry that ends at or
  after the offending line. A heading line inside an open fence still delimits blocks but gets
  ``entry_inside_code_fence`` (blocking for that identifier).
- Raw HTML. The first ``<`` followed by an ASCII letter, ``/``, ``!`` or ``?`` anywhere in the
  file (inside or outside an entry, in a fence, on a heading line) refuses every directive on that
  line and after it (``directive_after_raw_html``, blocking); directives above it still count, and
  ``parse_decision_entries`` reports it as ``raw_html_marker``. The scope is the rest of the
  file, not the entry, because raw HTML in one entry can hide later entries in a browser (an
  unclosed ``<!--`` is emitted before the entry's ``</li>``). As the log is append-only, one such
  byte pair refuses every later authority; the real log has none. A code span such as
  `` `<!--` `` is a known false refusal. ``]:`` anywhere (a link reference or footnote
  definition, whose possibly multi-line title is never rendered in place) refuses every later
  directive the same way (``directive_after_link_definition``, ``link_definition_marker``).
- Inline spans. A directive line without a bullet continues the paragraph above it, so a code
  span or strikethrough opened there could run over it: when an earlier line of the same
  paragraph (since the last blank line, list item line, fence line or column-0 line) holds a
  backtick or tilde, the bare directive is refused (``directive_in_inline_span``). A
  ``- authorizes: ...`` bullet starts a new block and is unaffected.
- A decision is closed when it appears in ``CLOSED_AUTHORITIES`` or when any block after it
  (decision or proposal; closing only removes authority) contains ``closes_authority: D-NNN``.
  Closure lines count everywhere, at any indentation, including in code, after raw HTML and
  outside any block. A closure line in the entry itself or before it is an ordering anomaly and
  also refuses. A line anywhere that starts like a closure (``closes_authority``, ``closes
  authority``, ``Closes-authority``, ``closes_authorty`` ... followed by ``:``) but is not in
  exact form closes every decision it names (``D-25``, trailing punctuation, wrong case); if it
  names no ``D-<digits>`` at all (``D025``, ``D 025``, an en dash, ``D-O25``) it is
  ``unattributable_closure_line`` and refuses every request. Not detected: a typo beside a
  well-formed identifier on the same line (``closes_authority: D-031, D025`` closes only D-031),
  a bold label (``**closes_authority:**``), a missing colon, and prose.
- One kind maps to one hash per entry: two different hashes for the same kind refuse.

Not modelled (stated so the claims stay exact): blockquote containers, setext headings, link
reference definitions and lazy continuation lines (a directive lazily merged into a quote or
paragraph is still visible text and counts); renderer extensions beyond CommonMark/GFM.

Known residual mis-attributions (independent review G5R, 2026-11-09; pre-existing, not
fixed): a bare directive line can still count while a renderer hides it, if it sits inside
(H1) an inline code span that runs across a list-marker-shaped line that cannot interrupt
a paragraph, (H2) a link reference definition with a multi-line label, (H3) multi-line
image alt text or link title, or (H4) a region split only by a lone CR. Each needs
deliberately unusual markup in a file only the owner edits; anyone able to edit the file
could add a valid line anyway. Writing rule for approvals: put each directive as its own
nested bullet directly under the decision heading, in an entry with no multi-line inline
code, link, image or HTML. A plain-text authority file would remove this class entirely.

Verification ceiling: L1 for every function (pure logic over supplied bytes, literal fixture
known answers and negative controls, fully verifiable on CPU). ``authenticity`` is always
``supplied_not_authenticated``: the verifier proves that the supplied bytes contain the line, not
that Huy wrote it, that the bytes are the committed file, or that the artifact bytes hash as
claimed. Callers hash the artifact and read the file themselves.

Never: reads or writes files, runs processes, touches the network or the clock, authenticates
provenance, authorizes from headings, prose, quotes or code spans, guesses a missing hash,
normalizes the case of a supplied hash, mutates its inputs, or raises on malformed or hostile
input (it returns issues instead). It decides nothing a consumer must decide:
``engineering_only_diagnostic`` is reported as a flag, not interpreted. It fills no owner value.
"""

import hashlib
import re

SCHEMA_VERSION = 1
AUTHENTICITY = "supplied_not_authenticated"
PRODUCER = "gate.s2_authority"
# D-019's bounded host check was completed and its allocation closed by D-020 on 2026-11-01.
CLOSED_AUTHORITIES = {"D-019": "D-020"}

_DECISION_ID = re.compile(r"D-[0-9]{3}")
_KIND = re.compile(r"[a-z][a-z0-9_]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REF = re.compile(r"[a-z][a-z0-9_]*(?::[a-z][a-z0-9_]*)?")
# ASCII digits only: ``\d`` would accept Arabic-Indic and fullwidth lookalikes of an identifier.
_ENTRY_HEADING = re.compile(r"- \*\*(D-[0-9]{3}) · ([0-9]{4}-[0-9]{2}-[0-9]{2}) · \S")
_PROPOSAL_HEADING = re.compile(r"- \*\*(P-[0-9]{3}) · \S")
# A block-ending list item that names a decision near its start (any marker and spacing, any
# emphasis, at most one leading word): '- __D-025', '- D-025', '- *D-031*', '- **Draft D-031'.
# Proposals are named only when emphasised, since the real log's '- P-001 · ...' items are
# plain proposal lines.
_LOOSE_HEADING = re.compile(r" ?(?:[-*+]|[0-9]{1,9}[.)])[ \t]+([*_]{0,3})"
                            r"(?:[A-Za-z]+[ \t]+[*_]{0,3})?([DP])-([0-9]*)")
_ENTRY_COLUMN = 2  # content column of a '- ' entry; continuation lines are indented at least this
_MAX_DIRECTIVE_INDENT = 5  # below an indented code block under any list item (2 + 4 = 6)
_LIST_MARKER = re.compile(r"([-*+]|[0-9]{1,9}[.)])([ \t]+|$)")
_FENCE_OPEN = re.compile(r"`{3,}|~{3,}")
_FENCE_CLOSE = re.compile(r"(`{3,}|~{3,})[ \t]*")
# '<' then a letter, '/' or '!' (tags, comments, declarations, CDATA) or '?' (a processing
# instruction, CommonMark HTML block type 3, which a browser reads as a bogus comment).
_HTML_MARK = re.compile(r"<[A-Za-z/!?]")
# A link reference or footnote definition ('[x]: /url "title' ... possibly over several lines)
# is not rendered where it stands; ']:' anywhere is treated as one, over-approximated.
_LINK_DEF_MARK = "]:"
# Characters that can open an inline code span or a strikethrough continuing onto later lines.
_SPAN_CHARS = ("`", "~")
_ITEM_START = re.compile(r"[ \t]*(?:[-*+]|[0-9]{1,9}[.)])(?:[ \t]|$)")
_PREFIX = r"[ \t]*(?:[-*+] )?"
_GRANT = re.compile(_PREFIX + r"authorizes: ([a-z][a-z0-9_]*) sha256:([0-9a-f]{64})[ \t]*")
_RECORD = re.compile(_PREFIX + r"records: ([a-z][a-z0-9_]*(?::[a-z][a-z0-9_]*)?)[ \t]*")
_FLAG = re.compile(_PREFIX + r"engineering_only_diagnostic: true[ \t]*")
_CLOSE = re.compile(r"[ \t]*(?:[-*+][ \t]+)?closes_authority: (D-[0-9]{3})[ \t]*")
# Lines that start like a directive but are not in exact form (typos refuse, never pass).
_NEAR = re.compile(r"[ \t+*-]*(?<!\*)"
                   r"(authorizes|records|engineering_only_diagnostic|closes_authority)\s*:",
                   re.IGNORECASE)
# Lines that start like a closure, including misspelled keywords (closes authority,
# closes_authorty).
_NEAR_CLOSE = re.compile(r"[ \t+*-]*(?<!\*)close[sd]?[ \t_-]?auth[a-z]*\s*:", re.IGNORECASE)
# Any decision reference a mistyped closure line could be attributed to (D-30, d-030, D-0025).
_ANY_DECISION_REF = re.compile(r"(?<![0-9A-Za-z])D-0*[0-9]{1,3}(?![0-9])", re.IGNORECASE)
_INTERNAL_KEYS = ("block_id", "from_line")


def _issue(code, message, *, severity="error", line=None):
    return {"code": code, "path": "docs/DECISIONS.md" + (f":{line}" if line else ""),
            "severity": severity, "evidence_refs": [], "message": message}


def _decode(decisions_md):
    """Return (raw, lines, issues): raw bytes (or None), CR-stripped str lines (or None)."""
    if not isinstance(decisions_md, (bytes, bytearray, memoryview)):
        return None, None, [_issue("decisions_not_bytes",
                                   "decisions_md must be bytes, got "
                                   f"{type(decisions_md).__name__}")]
    try:
        # Converted once. A released memoryview raises ValueError, and a bytes subclass may run
        # arbitrary __bytes__ code; any failure is a refusal, never an escaping exception.
        raw = bytes(decisions_md)
    except Exception as error:  # noqa: BLE001 - the never-raises contract covers hostile inputs
        return None, None, [_issue("decisions_not_bytes",
                                   f"decisions_md could not be read as bytes: {error}")]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        return raw, None, [_issue("decisions_not_utf8", f"decisions_md is not UTF-8: "
                                  f"{error.reason} at byte {error.start}")]
    lines = text.removeprefix("﻿").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return raw, [line.removesuffix("\r") for line in lines], []


def _advance(line, index, column):
    """Skip spaces/tabs from ``index`` at ``column``; return (column, index) after them."""
    while index < len(line) and line[index] in " \t":
        column = column + 1 if line[index] == " " else column + 4 - column % 4
        index += 1
    return column, index


def _indent(line):
    """Leading indentation in columns (a tab advances to the next multiple of 4)."""
    return _advance(line, 0, 0)[0]


def _irregular(number, reason, code="irregular_code_fence"):
    return _issue(code,
                  f"line {number} {reason}; rendered Markdown may place later lines inside or "
                  "outside a code block differently, so no entry ending at or after this line "
                  "can authorize", severity="warning", line=number) | {"from_line": number}


def _fence_opener(line, in_entry):
    """Return None, or ``(char, length, column, problem)`` for a line opening a code fence.

    The fence marker may follow any indentation and any list markers; ``problem`` names why the
    opener is irregular (a renderer may not read it as a fence), else None."""
    column, index = _advance(line, 0, 0)
    problem = None
    if "\t" in line[:index]:
        problem = "has a tab before its fence marker"
    elif column > (_MAX_DIRECTIVE_INDENT if in_entry else 3):
        problem = (f"opens a fence indented {column} columns, where it may render as indented "
                   "code or paragraph text")
    while (marker := _LIST_MARKER.match(line, index)) is not None and marker.group(2):
        if problem is None:
            problem = "opens a fence after a list marker"
        column, index = _advance(line, marker.end(1), column + marker.end(1) - index)
    mark = _FENCE_OPEN.match(line, index)
    if mark is None:
        return None
    run = mark.group(0)
    if problem is None and run[0] == "`" and "`" in line[mark.end():]:
        problem = "has a backtick in a backtick fence's info string, so it is not a fence"
    return run[0], len(run), column, problem


def _scan(lines):
    """Split lines into blocks and classify each directive line.

    Returns (blocks, orphans, issues, markers): ``markers`` maps ``after_html`` and
    ``after_link_definition`` to the first line holding that marker (or None)."""
    blocks, orphans, issues = [], [], []
    current = None
    fence = None  # open code region: {"char", "length", "column", "line"}
    markers = {"after_html": None, "after_link_definition": None}  # first line of each
    span_risk = False  # an earlier line of the current paragraph holds a backtick or tilde
    for number, line in enumerate(lines, start=1):
        indent = _indent(line)
        blank = line.strip(" \t") == ""
        if markers["after_html"] is None and _HTML_MARK.search(line):
            markers["after_html"] = number
        if markers["after_link_definition"] is None and _LINK_DEF_MARK in line:
            markers["after_link_definition"] = number
        # A list item line starts a new block, so no inline span reaches into it.
        if blank or indent < _ENTRY_COLUMN or _ITEM_START.match(line):
            span_risk = False
        if fence is not None and not blank and indent < fence["column"]:
            issues.append(_irregular(number, f"is less indented than the code fence opened at "
                                             f"line {fence['line']}"))
        if not blank and indent < _ENTRY_COLUMN:
            current = None
            heading = _ENTRY_HEADING.match(line) or _PROPOSAL_HEADING.match(line)
            loose = None if heading else _LOOSE_HEADING.match(line)
            named = heading is not None or (loose is not None
                                            and (loose.group(2) == "D" or loose.group(1)))
            block_id = (heading.group(1) if heading else
                        f"{loose.group(2)}-{loose.group(3)}" if loose and loose.group(3)
                        else None)
            if heading is not None:
                current = {"id": block_id,
                           "type": "decision" if block_id.startswith("D") else "proposal",
                           "start_line": number, "lines": [], "grants": [], "records": [],
                           "flag_lines": [], "closes": [], "near": [], "fenced": []}
                blocks.append(current)
            elif named:
                issues.append(_issue(
                    "malformed_decision_heading",
                    f"line {number} names a decision or proposal at column 0 but is not of the "
                    "form '- **D-NNN · YYYY-MM-DD · ...' or '- **P-NNN · ...'; it ends the "
                    "previous entry and is not an entry", severity="warning", line=number)
                    | {"block_id": block_id})
            if named and fence is not None:
                issues.append(_issue(
                    "entry_inside_code_fence",
                    f"line {number} heads a decision/proposal item inside an open code fence; it "
                    "is draft text, not a recorded entry", severity="warning", line=number)
                    | {"block_id": block_id})
        if current is not None:
            current["lines"].append(line)
        masked = None
        if fence is not None:
            masked = "fenced"
            closer = _FENCE_CLOSE.fullmatch(line.lstrip(" \t"))
            if (closer and closer.group(1)[0] == fence["char"]
                    and len(closer.group(1)) >= fence["length"]):
                if indent != fence["column"]:
                    issues.append(_irregular(number, f"closes the code fence opened at line "
                                                     f"{fence['line']} at another indentation"))
                fence = None
        else:
            opener = _fence_opener(line, current is not None)
            if opener is not None:
                char, length, column, problem = opener
                masked = "fenced"
                fence = {"char": char, "length": length, "column": column, "line": number}
                if problem is not None:
                    issues.append(_irregular(number, problem))
        for kind, first in markers.items():
            if masked is None and first is not None:
                masked = kind
        if masked is None and indent > _MAX_DIRECTIVE_INDENT:
            masked = "indented_code"
        if masked is None and span_risk:
            masked = "inline_span"  # a paragraph continuation line inside a possible span
        span_risk = (not blank and masked != "fenced"
                     and (span_risk or any(char in line for char in _SPAN_CHARS)))
        close = _CLOSE.fullmatch(line)
        if close:
            target = current if current is not None else {"id": None}
            record = {"line": number, "text": line.strip(), "target": close.group(1),
                      "block_id": target["id"]}
            (current["closes"] if current is not None else orphans).append(record)
            continue
        grant, ref, flag = _GRANT.fullmatch(line), _RECORD.fullmatch(line), _FLAG.fullmatch(line)
        directive = grant or ref or flag
        near_close = None if directive else _NEAR_CLOSE.match(line)
        near = None if directive else (_NEAR.match(line) or near_close)
        if directive is None and near is None:
            continue
        item = {"line": number, "text": line.strip()}
        if near_close is not None:
            # A mistyped closure anywhere must not leave an authority open (checked per target).
            orphans.append(item | {"kind": "near_close",
                                   "block_id": current["id"] if current is not None else None})
        if current is None:
            orphans.append(item | {"kind": "fenced" if masked == "fenced" else "outside_entry"})
        elif masked is not None:
            current["fenced"].append(item | {"kind": masked})
        elif grant:
            current["grants"].append(item | {"artifact_kind": grant.group(1),
                                              "sha256": grant.group(2)})
        elif ref:
            current["records"].append(item | {"ref": ref.group(1)})
        elif flag:
            current["flag_lines"].append(item)
        else:
            current["near"].append(item)
    if fence is not None:
        issues.append(_issue("unterminated_code_fence",
                             f"the code fence opened at line {fence['line']} is never closed; "
                             "lines after it cannot grant and no entry ending at or after it "
                             "can authorize", severity="warning", line=fence["line"])
                      | {"from_line": fence["line"]})
    return blocks, orphans, issues, markers


def _entry_sha256(lines):
    body = list(lines)
    while body and body[-1].strip() == "":
        body.pop()
    return hashlib.sha256(("\n".join(body) + "\n").encode("utf-8")).hexdigest()


def parse_decision_entries(decisions_md: bytes) -> dict:
    """Parse decision entries from the bytes of a decision log (L1; never raises).

    Returns ``{artifact_kind: 's2_decision_entries', schema_version, decisions_sha256,
    entries: {id: [occurrence, ...]}, proposals: [ids], issues}``. Each occurrence carries
    ``start_line``, ``date``, ``entry_sha256`` and the exact-form ``authorizes`` (kind, sha256,
    line), ``records``, ``engineering_only_diagnostic`` and ``closes_authority`` lines. A list with
    more than one occurrence means a duplicated identifier; ``verify_authority`` refuses it.
    """
    raw, lines, issues = _decode(decisions_md)
    digest = hashlib.sha256(raw).hexdigest() if raw is not None else None
    result = {"artifact_kind": "s2_decision_entries", "schema_version": SCHEMA_VERSION,
              "producer": PRODUCER, "decisions_sha256": digest, "entries": {},
              "proposals": [], "issues": issues}
    if lines is None:
        return result
    blocks, orphans, scan_issues, markers = _scan(lines)
    result["issues"] = issues + [{k: v for k, v in issue.items() if k not in _INTERNAL_KEYS}
                                 for issue in scan_issues]
    for kind, (code, what) in _MARKERS.items():
        if markers[kind] is not None:
            result["issues"].append(_issue(
                code, f"line {markers[kind]} holds {what}; every directive on or after that "
                "line is refused", severity="warning", line=markers[kind]))
    for block in blocks:
        if block["type"] == "proposal":
            result["proposals"].append(block["id"])
            continue
        heading = _ENTRY_HEADING.match(block["lines"][0])
        result["entries"].setdefault(block["id"], []).append({
            "start_line": block["start_line"], "date": heading.group(2),
            "entry_sha256": _entry_sha256(block["lines"]),
            "authorizes": [{"artifact_kind": g["artifact_kind"], "sha256": g["sha256"],
                            "line": g["line"]} for g in block["grants"]],
            "records": [{"ref": r["ref"], "line": r["line"]} for r in block["records"]],
            "engineering_only_diagnostic": bool(block["flag_lines"]),
            "closes_authority": [{"target": c["target"], "line": c["line"]}
                                 for c in block["closes"]],
        })
    for entry_id, occurrences in sorted(result["entries"].items()):
        if len(occurrences) > 1:
            result["issues"].append(_issue(
                "duplicate_decision_id",
                f"{entry_id} heads {len(occurrences)} entries (lines "
                f"{', '.join(str(o['start_line']) for o in occurrences)})", severity="warning"))
    for orphan in orphans:
        if "target" in orphan:
            continue
        if orphan.get("kind") == "near_close":
            result["issues"].append(_issue(
                "malformed_closure_line",
                f"line {orphan['line']} starts like closes_authority but is not in exact form; "
                "any decision it names is treated as closed", severity="warning",
                line=orphan["line"]))
            continue
        code, where, _severity = _MASKED[orphan["kind"]]
        result["issues"].append(_issue(code, f"line {orphan['line']} is {where}; it grants nothing",
                                       severity="warning", line=orphan["line"]))
    return result


# kind -> (code, where, severity inside the requested entry). A fenced directive is code and
# simply never counts; one after raw HTML or indented past the directive window is refused,
# because a renderer may show it as live text or hide it.
_MASKED = {"outside_entry": ("directive_outside_entry", "not inside a decision entry",
                             "warning"),
           "fenced": ("directive_in_code_fence", "inside a code fence", "warning"),
           "after_html": ("directive_after_raw_html",
                          "on or after the first raw HTML marker in the file", "error"),
           "after_link_definition": ("directive_after_link_definition",
                                     ("on or after the first link or footnote definition "
                                      "marker (']:') in the file"), "error"),
           "inline_span": ("directive_in_inline_span",
                           ("a paragraph continuation line after a backtick or tilde in the "
                            "same paragraph, where it may render inside a code span or "
                            "strikethrough"), "error"),
           "indented_code": ("directive_in_indented_code",
                             (f"indented more than {_MAX_DIRECTIVE_INDENT} columns, where it "
                              "may render as an indented code block"), "error")}


_MARKERS = {"after_html": ("raw_html_marker",
                           "a raw HTML marker ('<' then a letter, '/', '!' or '?')"),
            "after_link_definition": ("link_definition_marker",
                                      "a link or footnote definition marker (']:')")}


def _closure(blocks, orphans, index, decision_id):
    """Return (closed_by, issues) for blocks[index]; static closure first, then later lines."""
    issues = []
    closers = []
    for position, block in enumerate(blocks):
        for close in block["closes"]:
            if close["target"] == decision_id:
                closers.append((position, block["id"], close["line"]))
    for close in orphans:
        if "target" in close and close["target"] == decision_id:
            closers.append((None, None, close["line"]))
    named = re.compile(r"(?<![0-9A-Za-z])D-0*" + str(int(decision_id[2:])) + r"(?![0-9])",
                       re.IGNORECASE)
    for near in orphans:
        if near.get("kind") == "near_close" and named.search(near["text"]):
            issues.append(_issue(
                "malformed_closure_line",
                f"line {near['line']} is a mistyped closes_authority naming {decision_id}; "
                "treated as closed", line=near["line"]))
            closers.append((None, near["block_id"], near["line"]))
        elif near.get("kind") == "near_close" and not _ANY_DECISION_REF.search(near["text"]):
            issues.append(_issue(
                "unattributable_closure_line",
                f"line {near['line']} starts like closes_authority but names no D-<digits> "
                "identifier; it could be closing any decision, so nothing is authorized until it "
                "is corrected", line=near["line"]))
    start = blocks[index]["start_line"]
    closed_by = CLOSED_AUTHORITIES.get(decision_id)
    for position, block_id, line in sorted(closers, key=lambda c: c[2]):
        if position == index or line < start:
            issues.append(_issue(
                "closure_ordering_anomaly",
                f"closes_authority: {decision_id} at line {line} is not in a later block; "
                "treated as closed", line=line))
        if closed_by is None:
            closed_by = block_id if block_id is not None else f"line {line}"
    if closed_by is not None:
        issues.insert(0, _issue("authority_closed", f"{decision_id} is closed by {closed_by}"))
    return closed_by, issues


def _locate(decisions_md, decision_id):
    """Shared front half: decode, scan and find exactly one entry. Returns a context dict."""
    context = {"issues": [], "block": None, "blocks": [], "orphans": [], "index": None,
               "closed_by": None, "decisions_sha256": None}
    if not isinstance(decision_id, str) or not _DECISION_ID.fullmatch(decision_id):
        context["issues"].append(_issue("invalid_decision_id",
                                        f"decision_id must match D-NNN, got {decision_id!r}"))
    raw, lines, issues = _decode(decisions_md)
    if raw is not None:
        context["decisions_sha256"] = hashlib.sha256(raw).hexdigest()
    context["issues"].extend(issues)
    if lines is None or context["issues"]:
        return context
    blocks, orphans, scan_issues, _markers = _scan(lines)
    context["blocks"], context["orphans"] = blocks, orphans
    matches = [i for i, b in enumerate(blocks)
               if b["type"] == "decision" and b["id"] == decision_id]
    end_line = None
    if len(matches) == 1:
        found = blocks[matches[0]]
        end_line = found["start_line"] + len(found["lines"]) - 1
    for issue in scan_issues:
        block_id, from_line = issue.pop("block_id", None), issue.pop("from_line", None)
        # Issues naming the requested identifier, and fence irregularities at or before the end
        # of the requested entry (an irregular or unclosed fence), refuse; elsewhere they stay
        # warnings.
        if block_id == decision_id or (from_line is not None and end_line is not None
                                       and from_line <= end_line):
            issue["severity"] = "error"
        context["issues"].append(issue)
    if not matches:
        context["issues"].append(_issue("decision_not_found",
                                        f"no entry headed '- **{decision_id} · YYYY-MM-DD · ...'"))
        return context
    if len(matches) > 1:
        context["issues"].append(_issue(
            "duplicate_decision_id",
            f"{decision_id} heads {len(matches)} entries (lines "
            f"{', '.join(str(blocks[i]['start_line']) for i in matches)}); it must occur once"))
        return context
    context["index"] = matches[0]
    block = context["block"] = blocks[matches[0]]
    closed_by, closure_issues = _closure(blocks, orphans, matches[0], decision_id)
    context["closed_by"] = closed_by
    context["issues"].extend(closure_issues)
    for near in block["near"]:
        context["issues"].append(_issue(
            "malformed_directive_line",
            f"line {near['line']} in {decision_id} starts like a directive but is not in exact "
            "form", line=near["line"]))
    for fenced in block["fenced"]:
        code, where, severity = _MASKED[fenced["kind"]]
        context["issues"].append(_issue(
            code, f"line {fenced['line']} in {decision_id} is {where}; it grants nothing",
            severity=severity, line=fenced["line"]))
    return context


def _blocking(issues):
    return any(issue["severity"] == "error" for issue in issues)


def _valid_pair(kind, sha256):
    return (isinstance(kind, str) and _KIND.fullmatch(kind) is not None
            and isinstance(sha256, str) and _SHA256.fullmatch(sha256) is not None)


def verify_authority(decisions_md: bytes, *, decision_id: str, artifact_kind: str,
                     artifact_sha256: str, also_required: tuple = ()) -> dict:
    """Decide whether ``decision_id`` authorizes exactly this artifact hash (L1; never raises).

    ``authorized`` is true only when the entry occurs exactly once, is not closed, has no
    malformed directive line, binds ``artifact_kind`` to exactly one hash equal to
    ``artifact_sha256``, and binds every ``(kind, sha256)`` pair of ``also_required`` the same
    way. ``engineering_only_diagnostic`` reports the entry's flag line; consumers decide it.
    """
    result = {"artifact_kind": "s2_authority_check", "schema_version": SCHEMA_VERSION,
              "producer": PRODUCER, "authorized": False,
              "decision_id": decision_id if isinstance(decision_id, str) else None,
              "requested": [], "decisions_sha256": None, "entry_sha256": None,
              "matched_lines": [], "closed_by": None, "engineering_only_diagnostic": False,
              "issues": [], "authenticity": AUTHENTICITY}
    issues = result["issues"]
    requested = []
    if not _valid_pair(artifact_kind, artifact_sha256):
        issues.append(_issue(
            "invalid_artifact_request",
            "artifact_kind must match [a-z][a-z0-9_]* and artifact_sha256 must be 64 lowercase "
            f"hex characters; got {artifact_kind!r}, {artifact_sha256!r}"))
    else:
        requested.append((artifact_kind, artifact_sha256))
    if not isinstance(also_required, (tuple, list)):
        issues.append(_issue("invalid_also_required",
                             "also_required must be a tuple of (artifact_kind, sha256) pairs"))
        also_required = ()
    for pair in also_required:
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or not _valid_pair(pair[0], pair[1])):
            issues.append(_issue("invalid_also_required",
                                 f"also_required item {pair!r} is not a valid "
                                 "(artifact_kind, 64-lowercase-hex sha256) pair"))
        else:
            requested.append((pair[0], pair[1]))
    result["requested"] = [{"artifact_kind": k, "sha256": s} for k, s in requested]
    context = _locate(decisions_md, decision_id)
    issues.extend(context["issues"])
    block = context["block"]
    result["closed_by"] = context["closed_by"]
    result["decisions_sha256"] = context["decisions_sha256"]
    if block is None:
        return result
    result["entry_sha256"] = _entry_sha256(block["lines"])
    result["engineering_only_diagnostic"] = bool(block["flag_lines"])
    by_kind = {}
    for grant in block["grants"]:
        by_kind.setdefault(grant["artifact_kind"], []).append(grant)
    for kind, grants in sorted(by_kind.items()):
        if len({g["sha256"] for g in grants}) > 1:
            issues.append(_issue(
                "conflicting_authorizes_lines",
                f"{decision_id} binds {kind} to more than one hash (lines "
                f"{', '.join(str(g['line']) for g in grants)}); one kind authorizes one artifact"))
    if not block["grants"]:
        issues.append(_issue("no_authorizes_line",
                             f"{decision_id} contains no 'authorizes: <kind> sha256:<hex>' line"))
    matched = {}
    for kind, sha256 in requested:
        hits = [g for g in by_kind.get(kind, []) if g["sha256"] == sha256]
        if hits:
            for hit in hits:
                matched[hit["line"]] = hit["text"]
            continue
        if block["grants"]:
            bound = sorted({g["sha256"] for g in by_kind.get(kind, [])})
            issues.append(_issue(
                "artifact_not_authorized",
                f"{decision_id} has no line 'authorizes: {kind} sha256:{sha256}'"
                + (f"; it binds {kind} to {', '.join(bound)}" if bound else "")))
        others = sorted({b["id"] for i, b in enumerate(context["blocks"])
                         if i != context["index"]
                         and any(g["artifact_kind"] == kind and g["sha256"] == sha256
                                 for g in b["grants"])})
        if others:
            issues.append(_issue(
                "authorized_by_other_entry",
                f"'authorizes: {kind} sha256:{sha256}' appears in {', '.join(others)}, "
                f"not in {decision_id}"))
    result["matched_lines"] = [{"line": line, "text": text}
                               for line, text in sorted(matched.items())]
    result["authorized"] = bool(requested) and not _blocking(issues)
    return result


def verify_records(decisions_md: bytes, *, decision_id: str, refs: tuple) -> dict:
    """Decide whether ``decision_id`` records every bootstrap reference in ``refs`` (L1).

    A ref such as ``terms:harmonizer``, ``acquire:teacher`` or ``host_setup`` is satisfied only
    by a line ``records: <ref>`` in that single, unclosed entry. Empty ``refs`` is never satisfied.
    Never raises.
    """
    result = {"artifact_kind": "s2_records_check", "schema_version": SCHEMA_VERSION,
              "producer": PRODUCER, "satisfied": False,
              "decision_id": decision_id if isinstance(decision_id, str) else None,
              "refs": [], "decisions_sha256": None, "entry_sha256": None, "matched_lines": [],
              "missing_refs": [], "closed_by": None, "issues": [], "authenticity": AUTHENTICITY}
    issues = result["issues"]
    wanted = []
    if not isinstance(refs, (tuple, list)):
        issues.append(_issue("invalid_refs", "refs must be a tuple of reference strings"))
        refs = ()
    for ref in refs:
        if isinstance(ref, str) and _REF.fullmatch(ref):
            if ref not in wanted:
                wanted.append(ref)
        else:
            issues.append(_issue("invalid_ref", f"ref {ref!r} must match "
                                 "[a-z][a-z0-9_]*(:[a-z][a-z0-9_]*)?"))
    if not wanted and not issues:
        issues.append(_issue("no_refs", "refs is empty; nothing can be verified"))
    result["refs"] = sorted(wanted)
    context = _locate(decisions_md, decision_id)
    issues.extend(context["issues"])
    block = context["block"]
    result["closed_by"] = context["closed_by"]
    result["decisions_sha256"] = context["decisions_sha256"]
    if block is None:
        result["missing_refs"] = sorted(wanted)
        return result
    result["entry_sha256"] = _entry_sha256(block["lines"])
    present = {}
    for record in block["records"]:
        present.setdefault(record["ref"], []).append(record)
    matched, missing = [], []
    for ref in sorted(wanted):
        if ref in present:
            matched.extend(present[ref])
        else:
            missing.append(ref)
            issues.append(_issue("record_missing",
                                 f"{decision_id} has no line 'records: {ref}'"))
    result["matched_lines"] = [{"line": r["line"], "text": r["text"]}
                               for r in sorted(matched, key=lambda r: r["line"])]
    result["missing_refs"] = missing
    result["satisfied"] = bool(wanted) and not missing and not _blocking(issues)
    return result
