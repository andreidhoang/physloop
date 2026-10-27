#!/usr/bin/env python3
"""Guard PhysLoop's starred (★) cores: the files only Huy writes (CLAUDE.md, "Ownership").

Wired as Claude Code hooks in .claude/settings.json:
  PreToolUse  Edit|Write|MultiEdit|NotebookEdit  deny a write to a starred path.
  PreToolUse  Bash                               deny a command that names a starred path next to a
                                                 write verb; otherwise snapshot every starred file.
  PostToolUse(+Failure) Bash                     if the command changed a starred file anyway (cd, a
                                                 script, a glob), restore the snapshot and tell Claude.

Starred paths come from .claude/starred-cores.txt. The manifest, the hooks directory and the project
settings files are always protected, so an agent cannot switch the guard off from inside the repo.
Any internal error exits 2, which blocks the tool call: the guard fails closed.
"""
import base64
import json
import os
import re
import shlex
import sys
import tempfile
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2]).resolve()
MANIFEST = ROOT / ".claude" / "starred-cores.txt"
ALWAYS = [".claude/starred-cores.txt", ".claude/hooks/", ".claude/settings.json", ".claude/settings.local.json"]
FILE_TOOLS = {"Edit": "file_path", "Write": "file_path", "MultiEdit": "file_path", "NotebookEdit": "notebook_path"}
CACHE_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
GLOB_CHARS = "*?["

# Verbs that modify any path they name, and verbs that modify only their last argument.
ANY_ARG_WRITE = re.compile(
    r"\b(?:rm|mv|truncate|touch|tee|dd|patch|black|isort|autopep8|yapf)\b"
    r"|\bsed\s+(?:-\w+\s+)*-\w*i|\bperl\s+(?:-\w+\s+)*-\w*i"
    r"|\bgit\s+(?:checkout|restore|apply|mv|rm)\b|\bruff\s+format\b|--fix\b"
    r"|open\(|write_text|write_bytes|unlink|shutil\.|os\.(?:remove|rename|replace)"
)
DEST_WRITE = {"cp", "rsync", "install", "ln"}
SEGMENT_SPLIT = re.compile(r"\|\||&&|;|\||\n")


def entries():
    listed = []
    if MANIFEST.is_file():
        for line in MANIFEST.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                listed.append(line[2:] if line.startswith("./") else line)
    return ALWAYS + listed


def rel(path):
    p = Path(os.path.expanduser(path))
    if not p.is_absolute():
        p = ROOT / p
    try:
        return Path(os.path.normpath(p)).relative_to(ROOT).as_posix()
    except ValueError:
        return None


def is_starred(relpath, all_entries):
    for e in all_entries:
        if e.endswith("/"):
            if relpath == e[:-1] or relpath.startswith(e):
                return e
        elif any(c in e for c in GLOB_CHARS):
            if fnmatch(relpath, e):
                return e
        elif relpath == e:
            return e
    return None


def deny(reason):
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": reason}}))
    return 0


def reason_for(relpath, entry):
    return (f"★ {relpath} is a starred core (entry '{entry}' in .claude/starred-cores.txt): only Huy writes it. "
            "You may read it, explain the concept, write tests against it, or review it. "
            "To change it, describe the change and let Huy make it.")


def bash_mentions(segment, all_entries):
    for e in all_entries:
        if any(c in e for c in GLOB_CHARS):
            continue
        base = e.rstrip("/")
        for form in (base, (ROOT / base).as_posix()):
            if re.search(rf"(?<![\w.-]){re.escape(form)}(?![\w.-])", segment):
                return base, e
    return None


def bash_static_check(command, all_entries):
    for segment in SEGMENT_SPLIT.split(command):
        hit = bash_mentions(segment, all_entries)
        if not hit:
            continue
        relpath, entry = hit
        if re.search(rf">>?\s*['\"]?(?:{re.escape((ROOT / relpath).as_posix())}|(?:\./)?{re.escape(relpath)})", segment):
            return relpath, entry
        if ANY_ARG_WRITE.search(segment):
            return relpath, entry
        try:
            tokens = shlex.split(segment)
        except ValueError:
            return relpath, entry
        if tokens and os.path.basename(tokens[0]) in DEST_WRITE and bash_mentions(tokens[-1], all_entries):
            return relpath, entry
    return None


def starred_files(all_entries):
    files = {}
    for e in all_entries:
        if e.endswith("/"):
            d = ROOT / e[:-1]
            candidates = d.rglob("*") if d.is_dir() else []
        elif any(c in e for c in GLOB_CHARS):
            candidates = ROOT.glob(e)
        else:
            candidates = [ROOT / e]
        for f in candidates:
            if f.is_file() and not (CACHE_DIRS & set(f.parts)) and f.suffix != ".pyc":
                files[f.relative_to(ROOT).as_posix()] = f
    return files


def snapshot_path(data):
    key = re.sub(r"[^\w-]", "_", str(data.get("tool_use_id") or data.get("session_id") or "default"))
    d = Path(tempfile.gettempdir()) / "physloop-starred-guard"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def take_snapshot(data, all_entries):
    snap = {r: base64.b64encode(f.read_bytes()).decode() for r, f in starred_files(all_entries).items()}
    snapshot_path(data).write_text(json.dumps(snap))


def restore_after_bash(data, all_entries):
    """Undo starred-file changes made during the command. Nothing is destroyed: every displaced
    version is moved to .claude/starred-guard-backups/<stamp>/ first, in case it was Huy's own
    editor save that landed while the agent's command was running."""
    sp = snapshot_path(data)
    if not sp.is_file():
        return 0
    before = json.loads(sp.read_text())
    sp.unlink()
    now = starred_files(all_entries)
    backup = ROOT / ".claude" / "starred-guard-backups" / re.sub(r"[^\w-]", "_", str(data.get("tool_use_id") or "run"))
    changed = []

    def keep(r):
        src = ROOT / r
        if src.is_file():
            dst = backup / r
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.replace(dst)

    for r, b64 in before.items():
        f = ROOT / r
        original = base64.b64decode(b64)
        if not f.is_file() or f.read_bytes() != original:
            keep(r)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(original)
            changed.append(f"{r} (restored)")
    for r in now:
        if r not in before:
            keep(r)
            changed.append(f"{r} (created; moved to backup)")
    if not changed:
        return 0
    where = backup.relative_to(ROOT).as_posix()
    msg = ("That command changed starred cores, so the guard undid it: " + ", ".join(changed) + ". "
           "Starred cores are listed in .claude/starred-cores.txt and only Huy writes them. "
           "Describe the change you wanted and let Huy make it.")
    note = f"Starred-core guard reverted {', '.join(changed)}. Displaced versions are in {where}/ if any was your own edit."
    print(json.dumps({"decision": "block", "reason": msg, "systemMessage": note}))
    return 0


def main():
    data = json.load(sys.stdin)
    event, tool = data.get("hook_event_name", "PreToolUse"), data.get("tool_name", "")
    tool_input = data.get("tool_input") or {}
    all_entries = entries()
    if event == "PreToolUse" and tool in FILE_TOOLS:
        relpath = rel(tool_input.get(FILE_TOOLS[tool], ""))
        entry = relpath and is_starred(relpath, all_entries)
        return deny(reason_for(relpath, entry)) if entry else 0
    if tool == "Bash" and event == "PreToolUse":
        hit = bash_static_check(tool_input.get("command", ""), all_entries)
        if hit:
            return deny(reason_for(*hit))
        take_snapshot(data, all_entries)
        return 0
    if tool == "Bash" and event in ("PostToolUse", "PostToolUseFailure"):
        return restore_after_bash(data, all_entries)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # fail closed: exit 2 blocks the tool call
        print(f"starred-core guard error, blocking to be safe: {exc!r}", file=sys.stderr)
        sys.exit(2)
