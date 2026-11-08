"""Strict JSON identities and read-only artifact integrity checks for S2.

Hashes bind bytes and declared settings; they do not authenticate their origin,
prove physical validity, or grant scientific or execution admission.
"""

import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path


def _validate_json(value, active=None):
    """Reject Python extensions that JSON encoding would otherwise coerce."""
    if active is None:
        active = set()
    kind = type(value)
    if value is None or kind in (bool, int):
        return
    if kind is float:
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return
    if kind is str:
        value.encode("utf-8", errors="strict")
        return
    if kind not in (dict, list):
        raise ValueError(f"Expected a JSON-native value, got {kind.__name__}")
    marker = id(value)
    if marker in active:
        raise ValueError("JSON containers cannot be cyclic")
    active.add(marker)
    try:
        if kind is dict:
            for key, item in value.items():
                if type(key) is not str:
                    raise ValueError("JSON object keys must be strings")
                _validate_json(key, active)
                _validate_json(item, active)
        else:
            for item in value:
                _validate_json(item, active)
    finally:
        active.remove(marker)


def canonical_json_bytes(value):
    """Encode canonical-v1 JSON: strict UTF-8, sorted keys, compact separators."""
    try:
        _validate_json(value)
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds the supported depth") from exc


def identity_hash(value):
    """Return the SHA-256 of the canonical representation of a JSON value."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def read_json_bytes(data):
    """Parse UTF-8 JSON without duplicate keys, nonfinite numbers or coercions."""
    if type(data) is not bytes:
        raise ValueError("JSON input must be bytes")

    def pairs_to_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Non-JSON numeric constant: {value}")

    try:
        result = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs_to_object,
            parse_constant=reject_constant,
        )
        _validate_json(result)
        return result
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds the supported depth") from exc


def _directory_fd(root):
    """Open every root component without following symlinks (POSIX hosts)."""
    path = Path(root).absolute()
    if ".." in path.parts:
        raise ValueError("Artifact root cannot contain parent traversal")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    current = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            following = os.open(component, flags, dir_fd=current)
            os.close(current)
            current = following
        return current
    except BaseException:
        os.close(current)
        raise


def read_artifact(root, receipt):
    """Read one regular file under root and verify exactly the bytes returned.

    Receipt fields are exactly path, sha256 and size_bytes. Paths are relative
    POSIX component sequences with no empty, current or parent components.
    Descriptor-relative, no-follow opens prevent symlink substitution between a
    path check and opening the file. The caller parses these returned bytes.
    """
    if type(receipt) is not dict or set(receipt) != {"path", "sha256", "size_bytes"}:
        raise ValueError("Artifact receipt requires exactly path, sha256 and size_bytes")
    name, expected_hash, expected_size = (
        receipt["path"], receipt["sha256"], receipt["size_bytes"]
    )
    if type(name) is not str or not name or "\x00" in name:
        raise ValueError("Artifact path must be a nonempty relative POSIX path")
    components = name.split("/")
    if any(part in ("", ".", "..") for part in components):
        raise ValueError("Artifact path must be relative without traversal or empty components")
    if type(expected_hash) is not str or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
        raise ValueError("Artifact sha256 must be 64 lowercase hexadecimal characters")
    if type(expected_size) is not int or expected_size < 0:
        raise ValueError("Artifact size_bytes must be a nonnegative integer")

    directory = _directory_fd(root)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        for component in components[:-1]:
            following = os.open(component, directory_flags, dir_fd=directory)
            os.close(directory)
            directory = following
        descriptor = os.open(
            components[-1],
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
            dir_fd=directory,
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("Artifact must be a regular file")
            if metadata.st_size != expected_size:
                raise ValueError("Artifact size does not match receipt")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                data = stream.read(expected_size + 1)
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)
    if len(data) != expected_size:
        raise ValueError("Artifact size changed while reading")
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError("Artifact checksum does not match receipt")
    return data
