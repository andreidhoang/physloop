"""Torch-free safetensors tensor manifest and comparator (build-spec A6; CPU-05 F2).

Serves S2.4 [C] (G3·R1, G7): the frozen-teacher re-export control needs "identical
checkpoint tensor values" kept apart from "identical serialization files", and
S4 engineering preparation (A10/B3) hashes loaded torch tensors with the same
convention. Agent-authored evaluator engineering under D-018/D-024.

Tensor identity convention (build-spec section 2 rule 11): a tensor's identity
hash is SHA-256 over exactly the byte range a safetensors file stores for it (raw
bytes in the stored dtype, row-major contiguous, little-endian), with no dtype or
shape prefix; the dtype string and shape are recorded beside it. A hash computed
from a loaded tensor's contiguous little-endian bytes therefore equals the hash
of its stored byte range. File hashes, header hashes and ``__metadata__`` are
reported separately, so a re-export whose tensors are identical but whose header
key order, padding or metadata differ compares ``tensors_identical_files_differ``.

Verification level per function (build-spec section 1):

- ``tensor_sha256``, ``tensor_record``, ``parse_safetensors``, ``build_manifest``,
  ``compare_manifests`` - L1: pure functions over bytes/JSON with hand-written
  byte fixtures, literal hashes and negative controls.
- ``build_manifest_from_directory`` - L1 on CPU temporary directories (no-follow
  open, regular files only, mmap). Its use on an actual exported checkpoint is
  L3 (``qualify_on_host``): a CPU pass qualifies the comparator, not an export.

Format rules enforced (fail closed; stricter than upstream where noted):

- 8-byte little-endian header length ``N``; ``N == 0`` is rejected; ``N`` above
  ``max_header_bytes`` is rejected before any decoding; ``8 + N`` must fit.
- The header's first byte is ``{`` and it decodes with
  ``gate.s2_identity.read_json_bytes`` (strict UTF-8, duplicate keys and
  nonfinite constants rejected; trailing JSON whitespace, i.e. upstream's space
  padding, is accepted; a trailing NUL is rejected).
- Each tensor entry has exactly ``dtype``, ``shape``, ``data_offsets``; dtype is
  one of the twelve rule-11 names (``DTYPE_ITEMSIZE``); shape elements are
  non-negative ints (not booleans); offsets are two ints ``0 <= begin <= end``;
  ``end - begin == prod(shape) * itemsize`` (scalar ``[]`` has one element).
- Sorted by offset, the byte ranges start at 0, are contiguous (no hole, no
  overlap) and end exactly at the data length; header key order need not match
  offset order. ``__metadata__`` is a string-to-string map.
- Sharded checkpoints: a ``model.safetensors.index.json``-style index with
  exactly ``weight_map`` (tensor name -> bare shard file name) and optional
  ``metadata``; every indexed shard is present, every present shard is indexed,
  and each tensor lives in exactly the one shard the index names. A declared
  ``metadata.total_size`` that differs from the summed tensor bytes is a
  warning (it does not change tensor identity).
- A manifest whose files hold no tensors at all is blocked (``empty_tensor_set``):
  absence is never a pass, so two empty checkpoints never compare identical.
- File identity in ``compare_manifests`` covers every shard and the shard index.
- ``max_header_bytes`` is validated before any file is opened; the directory
  reader closes every mapping and descriptor even when an exception's traceback
  still holds a view of a mapping.

Observed against the safetensors 0.7.0 reader on 2026-11-08 (agent scratch probe,
not a shipped test): it rejects a header length of 100,000,001 ("header too
large"), holes, overlaps, trailing uncovered data and a NUL after the JSON; it
accepts header key order different from offset order. It also accepts duplicate
keys, unknown tensor-entry keys, a leading space before ``{`` and the dtype
``U16``, all of which this module rejects deliberately.

Agent technical defaults (D-018): ``MAX_HEADER_BYTES = 100_000_000`` (equal to
the upstream reader limit observed above); ``max_file_bytes=None`` (no file cap)
for the directory reader, which maps files instead of loading them into RAM.

This module never imports torch or safetensors, never executes model code,
never downloads, never writes files, never follows symlinks when reading
evidence and never decides whether a difference is scientifically acceptable.
"""

import hashlib
import math
import mmap
import os
import re
import stat
import unicodedata
from pathlib import Path

from gate.s2_identity import identity_hash, read_json_bytes

ARTIFACT_KIND = "tensor_manifest"
COMPARISON_KIND = "tensor_manifest_comparison"
SCHEMA_VERSION = 1
SCOPES = ("synthetic", "public_source_fixture", "actual")
MAX_HEADER_BYTES = 100_000_000
DTYPE_ITEMSIZE = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}
HASH_CONVENTION = (
    "sha256 over the stored little-endian row-major byte range of each tensor, no "
    "dtype/shape prefix; dtype and shape recorded separately (build-spec section 2 rule 11)"
)
UNITS = {
    "byte_length": "bytes",
    "size_bytes": "bytes",
    "header_bytes": "bytes",
    "tensor_bytes": "bytes",
    "shape": "elements per dimension, row-major",
}

_TENSOR_KEYS = {"dtype", "shape", "data_offsets"}
_ENTRY_KEYS = {"file", "dtype", "shape", "byte_length", "sha256"}
_MANIFEST_KEYS = {
    "artifact_kind",
    "schema_version",
    "scope",
    "producer",
    "input_hashes",
    "units",
    "hash_convention",
    "status",
    "layout",
    "files",
    "index",
    "tensors",
    "tensor_set_sha256",
    "totals",
    "issues",
}
_HEX64 = re.compile(r"[0-9a-f]{64}")
_WHITESPACE = b" \t\r\n"


class TensorManifestError(ValueError):
    """A safetensors file or tensor record that fails closed, with a stable code."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _issue(code, path, message, severity="error"):
    return {
        "code": code,
        "path": path,
        "severity": severity,
        "evidence_refs": [],
        "message": message,
    }


def _check_scope(scope):
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")


def _check_header_cap(max_header_bytes):
    if type(max_header_bytes) is not int or max_header_bytes < 1:
        raise ValueError("max_header_bytes must be a positive integer")


def _is_count(value):
    return type(value) is int and value >= 0


def _shape_numel(dtype, shape):
    if dtype not in DTYPE_ITEMSIZE:
        raise TensorManifestError("unsupported_dtype", f"Unsupported dtype {dtype!r}")
    if type(shape) is not list or not all(_is_count(dim) for dim in shape):
        raise TensorManifestError(
            "tensor_shape_invalid", "Shape must be a list of non-negative integers"
        )
    return math.prod(shape)


def tensor_sha256(raw):
    """Return the rule-11 identity hash of a tensor's raw stored bytes (L1)."""
    return hashlib.sha256(memoryview(raw)).hexdigest()


def tensor_record(dtype, shape, raw):
    """Return ``{dtype, shape, byte_length, sha256}`` for raw little-endian bytes (L1).

    ``raw`` must be the row-major contiguous little-endian bytes of the tensor in
    its stored dtype, exactly ``prod(shape) * itemsize`` bytes long.
    """
    view = memoryview(raw).cast("B")
    numel = _shape_numel(dtype, shape)
    if view.nbytes != numel * DTYPE_ITEMSIZE[dtype]:
        raise TensorManifestError(
            "tensor_size_mismatch",
            f"{view.nbytes} bytes do not match {dtype} shape {shape}",
        )
    return {
        "dtype": dtype,
        "shape": list(shape),
        "byte_length": view.nbytes,
        "sha256": tensor_sha256(view),
    }


def parse_safetensors(data, *, max_header_bytes=MAX_HEADER_BYTES):
    """Parse and validate a safetensors buffer without decoding tensor values (L1).

    Returns ``header_bytes``, ``header_sha256``, ``metadata`` (``None`` when
    absent) and ``tensors`` (name -> dtype, shape, data_offsets, byte_length).
    Raises ``TensorManifestError`` with a stable ``code`` on any violation.
    """
    _check_header_cap(max_header_bytes)
    view = memoryview(data).cast("B")
    total = view.nbytes
    if total < 8:
        raise TensorManifestError("file_too_short", "File is shorter than the 8-byte prefix")
    header_length = int.from_bytes(view[:8], "little")
    if header_length == 0:
        raise TensorManifestError("header_empty", "Header length is zero")
    if header_length > max_header_bytes:
        raise TensorManifestError(
            "header_too_large", f"Header length {header_length} exceeds {max_header_bytes}"
        )
    if 8 + header_length > total:
        raise TensorManifestError(
            "header_length_out_of_bounds", "Header length exceeds the file size"
        )
    header = bytes(view[8 : 8 + header_length])
    if header[:1] != b"{":
        raise TensorManifestError("header_start_invalid", "Header must start with '{'")
    try:
        decoded = read_json_bytes(header)
    except ValueError as exc:
        raise TensorManifestError("header_json_invalid", f"Header JSON invalid: {exc}") from exc
    if type(decoded) is not dict:
        raise TensorManifestError("header_start_invalid", "Header must be a JSON object")

    data_length = total - 8 - header_length
    metadata = None
    tensors = {}
    for name, entry in decoded.items():
        if name == "__metadata__":
            if type(entry) is not dict or not all(type(v) is str for v in entry.values()):
                raise TensorManifestError(
                    "metadata_invalid", "__metadata__ must map strings to strings"
                )
            metadata = dict(sorted(entry.items()))
            continue
        if not name or "\x00" in name:
            raise TensorManifestError("tensor_name_invalid", "Tensor names must be nonempty")
        if type(entry) is not dict or set(entry) != _TENSOR_KEYS:
            raise TensorManifestError(
                "tensor_entry_invalid",
                f"Tensor {name!r} must have exactly dtype, shape and data_offsets",
            )
        dtype, shape, offsets = entry["dtype"], entry["shape"], entry["data_offsets"]
        numel = _shape_numel(dtype, shape)
        if (
            type(offsets) is not list
            or len(offsets) != 2
            or not all(_is_count(v) for v in offsets)
            or offsets[0] > offsets[1]
        ):
            raise TensorManifestError(
                "tensor_offsets_invalid", f"Tensor {name!r} offsets must be [begin, end]"
            )
        begin, end = offsets
        if end - begin != numel * DTYPE_ITEMSIZE[dtype]:
            raise TensorManifestError(
                "tensor_size_mismatch",
                f"Tensor {name!r} byte range does not match {dtype} shape {shape}",
            )
        if end > data_length:
            raise TensorManifestError(
                "tensor_offsets_out_of_bounds", f"Tensor {name!r} ends beyond the data"
            )
        tensors[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [begin, end],
            "byte_length": end - begin,
        }

    expected = 0
    for name, entry in sorted(tensors.items(), key=lambda kv: (*kv[1]["data_offsets"], kv[0])):
        begin, end = entry["data_offsets"]
        if begin != expected:
            raise TensorManifestError(
                "data_not_contiguous",
                f"Tensor {name!r} starts at {begin}, expected {expected} (hole or overlap)",
            )
        expected = end
    if expected != data_length:
        raise TensorManifestError(
            "data_not_fully_covered",
            f"Tensors cover {expected} of {data_length} data bytes",
        )
    return {
        "header_bytes": header_length,
        "header_sha256": hashlib.sha256(header).hexdigest(),
        "metadata": metadata,
        "tensors": dict(sorted(tensors.items())),
    }


def _check_name(name, *, bare=False):
    if type(name) is not str or not name or "\x00" in name or "\\" in name:
        return False
    parts = name.split("/")
    if bare and len(parts) != 1:
        return False
    return not any(part in ("", ".", "..") for part in parts)


def _check_file_names(names):
    if not names:
        raise ValueError("At least one file is required")
    seen = {}
    for name in names:
        if not _check_name(name):
            raise ValueError(f"File name must be a relative POSIX path without traversal: {name!r}")
        key = unicodedata.normalize("NFC", name).casefold()
        if key in seen:
            raise ValueError(f"Duplicate normalized file names: {seen[key]!r} and {name!r}")
        seen[key] = name


def _parse_index(raw):
    """Return (weight_map, declared_total_size) or raise ValueError."""
    decoded = read_json_bytes(bytes(raw))
    if type(decoded) is not dict or "weight_map" not in decoded:
        raise ValueError("Index must be an object with weight_map")
    if not set(decoded) <= {"weight_map", "metadata"}:
        raise ValueError("Index allows only weight_map and metadata")
    weight_map = decoded["weight_map"]
    if type(weight_map) is not dict or not weight_map:
        raise ValueError("weight_map must be a nonempty object")
    for tensor, shard in weight_map.items():
        if not tensor or not _check_name(shard, bare=True):
            raise ValueError(f"weight_map entry {tensor!r} must name a bare shard file")
    total = None
    metadata = decoded.get("metadata", {})
    if type(metadata) is not dict:
        raise ValueError("Index metadata must be an object")
    if "total_size" in metadata:
        total = metadata["total_size"]
        if not _is_count(total):
            raise ValueError("metadata.total_size must be a non-negative integer")
    return weight_map, total


def build_manifest(files, *, scope, index_name=None, max_header_bytes=MAX_HEADER_BYTES):
    """Build a tensor manifest from ``{file name: bytes-like}`` (L1).

    ``index_name`` names the shard index among ``files`` (a bare name: shard
    names in its ``weight_map`` are resolved beside it); every other file is a
    safetensors shard. A malformed file or index is reported as an error issue
    (status ``blocked``) rather than raised; caller errors (bad scope, bad or
    colliding names, an index name absent from ``files``) raise ``ValueError``.
    """
    _check_scope(scope)
    _check_header_cap(max_header_bytes)
    _check_file_names(list(files))
    if index_name is not None and (index_name not in files or not _check_name(index_name, bare=True)):
        raise ValueError("index_name must be a bare file name among the supplied files")
    issues = []
    file_entries = []
    input_hashes = {}
    owner = {}
    tensors = {}
    duplicates = set()
    parsed_ok = {}
    for name in sorted(files):
        # Views are released on exit, so a caller's mmap can be closed afterwards.
        with memoryview(files[name]) as supplied, supplied.cast("B") as view:
            input_hashes[name] = hashlib.sha256(view).hexdigest()
            if name == index_name:
                continue
            entry = {
                "name": name,
                "sha256": input_hashes[name],
                "size_bytes": view.nbytes,
                "parsed": False,
                "header_bytes": None,
                "header_sha256": None,
                "metadata": None,
                "tensor_count": None,
            }
            file_entries.append(entry)
            try:
                parsed = parse_safetensors(view, max_header_bytes=max_header_bytes)
            except TensorManifestError as exc:
                issues.append(_issue(exc.code, f"files/{name}", str(exc)))
                parsed_ok[name] = None
                continue
            parsed_ok[name] = parsed
            entry.update(
                parsed=True,
                header_bytes=parsed["header_bytes"],
                header_sha256=parsed["header_sha256"],
                metadata=parsed["metadata"],
                tensor_count=len(parsed["tensors"]),
            )
            data_start = 8 + parsed["header_bytes"]
            for tensor, info in parsed["tensors"].items():
                if tensor in owner:
                    duplicates.add(tensor)
                    issues.append(
                        _issue(
                            "tensor_in_multiple_files",
                            f"tensors/{tensor}",
                            f"Tensor {tensor!r} appears in {owner[tensor]!r} and {name!r}",
                        )
                    )
                    continue
                owner[tensor] = name
                begin, end = info["data_offsets"]
                tensors[tensor] = {
                    "file": name,
                    "dtype": info["dtype"],
                    "shape": info["shape"],
                    "byte_length": info["byte_length"],
                    "sha256": tensor_sha256(view[data_start + begin : data_start + end]),
                }
    for tensor in duplicates:
        tensors.pop(tensor, None)

    index_section = None
    if index_name is not None:
        index_section = {
            "name": index_name,
            "sha256": input_hashes[index_name],
            "size_bytes": memoryview(files[index_name]).nbytes,
            "total_size_declared": None,
            "total_size_observed": None,
        }
        try:
            weight_map, declared = _parse_index(files[index_name])
        except ValueError as exc:
            issues.append(_issue("index_invalid", f"index/{index_name}", str(exc)))
        else:
            index_section["total_size_declared"] = declared
            shard_names = set(parsed_ok)
            for shard in sorted(set(weight_map.values()) - shard_names):
                issues.append(
                    _issue("index_shard_missing", f"index/{shard}", "Indexed shard not supplied")
                )
            for shard in sorted(shard_names - set(weight_map.values())):
                issues.append(
                    _issue("file_not_in_index", f"files/{shard}", "Shard is not in the index")
                )
            for shard, parsed in sorted(parsed_ok.items()):
                for tensor in parsed["tensors"] if parsed else ():
                    if tensor not in weight_map:
                        issues.append(
                            _issue(
                                "tensor_not_in_index",
                                f"tensors/{tensor}",
                                f"Tensor in {shard!r} is absent from the index",
                            )
                        )
                    elif weight_map[tensor] != shard:
                        issues.append(
                            _issue(
                                "index_shard_mismatch",
                                f"tensors/{tensor}",
                                f"Index names {weight_map[tensor]!r}; tensor found in {shard!r}",
                            )
                        )
            for tensor, shard in sorted(weight_map.items()):
                parsed = parsed_ok.get(shard)
                if parsed is not None and tensor not in parsed["tensors"]:
                    issues.append(
                        _issue(
                            "indexed_tensor_missing",
                            f"tensors/{tensor}",
                            f"Indexed shard {shard!r} lacks the tensor",
                        )
                    )
            if all(parsed_ok.values()):
                observed = sum(
                    info["byte_length"]
                    for parsed in parsed_ok.values()
                    for info in parsed["tensors"].values()
                )
                index_section["total_size_observed"] = observed
                if declared is not None and declared != observed:
                    issues.append(
                        _issue(
                            "index_total_size_mismatch",
                            f"index/{index_name}",
                            f"metadata.total_size {declared} differs from {observed} tensor bytes",
                            severity="warning",
                        )
                    )

    if not tensors and not any(item["severity"] == "error" for item in issues):
        # Absence is never a pass: two empty manifests would otherwise compare identical.
        issues.append(
            _issue("empty_tensor_set", "tensors", "The supplied files hold no tensors")
        )
    issues.sort(key=lambda item: (item["path"], item["code"], item["message"]))
    blocked = any(item["severity"] == "error" for item in issues)
    tensors = dict(sorted(tensors.items()))
    if index_name is not None:
        layout = "sharded"
    elif len(file_entries) == 1:
        layout = "single_file"
    else:
        layout = "multiple_files"
    return {
        "artifact_kind": ARTIFACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "producer": {"module": "gate.tensor_manifest", "source_sha256": _source_sha256()},
        "input_hashes": {"files": input_hashes},
        "units": dict(UNITS),
        "hash_convention": HASH_CONVENTION,
        "status": "blocked" if blocked else "passed",
        "layout": layout,
        "files": file_entries,
        "index": index_section,
        "tensors": tensors,
        "tensor_set_sha256": None if blocked else _tensor_set_sha256(tensors),
        "totals": {
            "tensor_count": len(tensors),
            "tensor_bytes": sum(item["byte_length"] for item in tensors.values()),
            "file_count": len(file_entries),
        },
        "issues": issues,
    }




def _tensor_set_sha256(tensors):
    """Identity of the tensor set alone: names, dtypes, shapes and byte hashes."""
    return identity_hash(
        {
            name: {key: item[key] for key in ("dtype", "shape", "byte_length", "sha256")}
            for name, item in tensors.items()
        }
    )


def _open_root(root):
    path = Path(root).absolute()
    if ".." in path.parts:
        raise ValueError("Checkpoint root cannot contain parent traversal")
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


def _open_regular(root_fd, name):
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    parts = name.split("/")
    directory = os.dup(root_fd)
    try:
        for component in parts[:-1]:
            following = os.open(component, flags, dir_fd=directory)
            os.close(directory)
            directory = following
        return os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=directory
        )
    finally:
        os.close(directory)


def build_manifest_from_directory(
    root,
    file_names,
    *,
    scope,
    index_name=None,
    max_header_bytes=MAX_HEADER_BYTES,
    max_file_bytes=None,
):
    """Read regular files under ``root`` without following symlinks and build a manifest.

    ``file_names`` lists shard paths relative to ``root``; ``None`` is allowed
    only with ``index_name``, whose ``weight_map`` then names the shards. Files
    are memory-mapped read-only (no whole-file RAM copy); identity, size and
    modification time are checked before and after so a file changed while
    being read is rejected. Every filesystem failure raises ``ValueError``.
    """
    _check_scope(scope)
    _check_header_cap(max_header_bytes)  # caller errors are raised before anything is opened
    if max_file_bytes is not None and (type(max_file_bytes) is not int or max_file_bytes < 0):
        raise ValueError("max_file_bytes must be a non-negative integer or None")
    if index_name is not None and not _check_name(index_name, bare=True):
        raise ValueError("index_name must be a bare file name at the checkpoint root")
    try:
        root_fd = _open_root(root)
    except OSError as exc:
        raise ValueError(f"Cannot open checkpoint root without following links: {exc}") from exc
    descriptors = {}
    buffers = {}
    try:

        def load(name):
            try:
                descriptor = _open_regular(root_fd, name)
            except OSError as exc:
                raise ValueError(f"Cannot open {name!r} as a regular no-follow file: {exc}") from exc
            descriptors[name] = descriptor
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"{name!r} is not a regular file")
            if max_file_bytes is not None and before.st_size > max_file_bytes:
                raise ValueError(f"{name!r} exceeds max_file_bytes")
            if before.st_size:
                buffers[name] = mmap.mmap(descriptor, before.st_size, access=mmap.ACCESS_READ)
            else:
                buffers[name] = b""
            return before

        if index_name is not None:
            _check_file_names([index_name])
            load(index_name)
            if file_names is None:
                try:
                    weight_map, _ = _parse_index(buffers[index_name])
                except ValueError as exc:
                    raise ValueError(f"Index {index_name!r} is invalid: {exc}") from exc
                file_names = sorted(set(weight_map.values()))
        elif file_names is None:
            raise ValueError("file_names may be None only when index_name is given")
        _check_file_names(list(file_names) + ([index_name] if index_name else []))
        stats = {}
        if index_name is not None:
            stats[index_name] = os.fstat(descriptors[index_name])
        for name in file_names:
            stats[name] = load(name)
        report = build_manifest(
            buffers, scope=scope, index_name=index_name, max_header_bytes=max_header_bytes
        )
        for name, before in stats.items():
            after = os.fstat(descriptors[name])
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError(f"{name!r} changed while being read")
    except BaseException:
        _release(buffers, descriptors, root_fd, strict=False)
        raise
    _release(buffers, descriptors, root_fd, strict=True)
    return report


def _release(buffers, descriptors, root_fd, *, strict):
    """Close every mapping and descriptor, each in its own try block.

    While an exception propagates, its traceback may still hold a view of a
    mapping, so ``mmap.close`` raises ``BufferError``; that mapping is unmapped
    when the traceback is freed, and every descriptor is still closed here. On
    the success path (``strict``) any close failure is raised, never swallowed.
    """
    failures = []
    for name, buffer in buffers.items():
        if isinstance(buffer, mmap.mmap):
            try:
                buffer.close()
            except BufferError as exc:
                failures.append(f"{name!r}: {exc}")
    for name, descriptor in descriptors.items():
        try:
            os.close(descriptor)
        except OSError as exc:
            failures.append(f"{name!r}: {exc}")
    try:
        os.close(root_fd)
    except OSError as exc:
        failures.append(f"root: {exc}")
    if strict and failures:
        raise ValueError("Could not release checkpoint files: " + "; ".join(failures))


def _manifest_problem(manifest):
    """Return (code, message) when a manifest cannot be compared, else None."""
    if (
        type(manifest) is not dict
        or set(manifest) != _MANIFEST_KEYS
        or manifest["artifact_kind"] != ARTIFACT_KIND
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["scope"] not in SCOPES
        or type(manifest["tensors"]) is not dict
        or type(manifest["files"]) is not list
    ):
        return "manifest_invalid", "Not a tensor_manifest schema_version 1 artifact"
    if manifest["status"] != "passed":
        return "manifest_blocked", "A blocked manifest cannot be compared"
    if not manifest["tensors"]:
        # As in build_manifest: absence is never a pass, whatever status the manifest claims.
        return "empty_tensor_set", "A manifest with no tensors cannot be compared"
    for name, entry in manifest["tensors"].items():
        try:
            if type(entry) is not dict or set(entry) != _ENTRY_KEYS:
                raise TensorManifestError("manifest_inconsistent", "bad entry keys")
            numel = _shape_numel(entry["dtype"], entry["shape"])
            if (
                entry["byte_length"] != numel * DTYPE_ITEMSIZE[entry["dtype"]]
                or type(entry["sha256"]) is not str
                or _HEX64.fullmatch(entry["sha256"]) is None
            ):
                raise TensorManifestError("manifest_inconsistent", "bad entry values")
        except TensorManifestError:
            return "manifest_inconsistent", f"Tensor entry {name!r} is inconsistent"
    for entry in manifest["files"]:
        if type(entry) is not dict or not {"name", "sha256", "metadata"} <= set(entry):
            return "manifest_inconsistent", "File entries need name, sha256 and metadata"
    index = manifest["index"]
    if index is not None and (type(index) is not dict or not {"name", "sha256"} <= set(index)):
        return "manifest_inconsistent", "The index entry needs name and sha256"
    try:
        recomputed = _tensor_set_sha256(manifest["tensors"])
    except ValueError:
        return "manifest_inconsistent", "Tensor entries are not JSON-native"
    if recomputed != manifest["tensor_set_sha256"]:
        return "manifest_inconsistent", "tensor_set_sha256 does not match the tensor entries"
    return None


def compare_manifests(a, b):
    """Compare two passed manifests by tensor identity and, separately, by file (L1).

    For each common tensor name the differing fields among ``dtype``, ``shape``
    and ``sha256`` are listed; ``bytes_changed`` means the stored byte hash
    differs. Same bytes under a different dtype or shape is *not* identical.
    ``tensors_identical`` is independent of file hashes, header order, padding
    and ``__metadata__``; ``files_identical`` compares every shard and the shard
    index by name and SHA-256. Blocked, tampered, unknown-version or cross-scope
    inputs give status ``blocked`` with every verdict field ``None``.
    """
    issues = []
    for label, manifest in (("a", a), ("b", b)):
        problem = _manifest_problem(manifest)
        if problem:
            issues.append(_issue(problem[0], label, problem[1]))
    if not issues and a["scope"] != b["scope"]:
        issues.append(_issue("scope_mismatch", "scope", "Manifests come from different scopes"))
    result = {
        "artifact_kind": COMPARISON_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": None,
        "producer": {"module": "gate.tensor_manifest", "source_sha256": _source_sha256()},
        "input_hashes": {"a": None, "b": None},
        "units": dict(UNITS),
        "hash_convention": HASH_CONVENTION,
        "status": "blocked",
        "tensors_identical": None,
        "files_identical": None,
        "metadata_identical": None,
        "relation": None,
        "identical": [],
        "only_in_a": [],
        "only_in_b": [],
        "changed": [],
        "dtype_changed": [],
        "shape_changed": [],
        "bytes_changed": [],
        "files": {"a": [], "b": []},
        "issues": issues,
    }
    for label, manifest in (("a", a), ("b", b)):
        try:
            result["input_hashes"][label] = identity_hash(manifest)
        except ValueError:
            result["input_hashes"][label] = None
    if issues:
        issues.sort(key=lambda item: (item["path"], item["code"]))
        return result

    ta, tb = a["tensors"], b["tensors"]
    for name in sorted(set(ta) & set(tb)):
        fields = [key for key in ("dtype", "shape", "sha256") if ta[name][key] != tb[name][key]]
        if not fields:
            result["identical"].append(name)
            continue
        view = ("dtype", "shape", "sha256")
        result["changed"].append(
            {
                "name": name,
                "fields": fields,
                "a": {key: ta[name][key] for key in view},
                "b": {key: tb[name][key] for key in view},
            }
        )
        for key, bucket in (("dtype", "dtype"), ("shape", "shape"), ("sha256", "bytes")):
            if key in fields:
                result[f"{bucket}_changed"].append(name)
    result["only_in_a"] = sorted(set(ta) - set(tb))
    result["only_in_b"] = sorted(set(tb) - set(ta))

    def file_view(manifest):
        # The shard index is a serialized file too: its bytes are part of file identity.
        entries = [{"name": f["name"], "sha256": f["sha256"]} for f in manifest["files"]]
        if manifest["index"] is not None:
            index = manifest["index"]
            entries.append({"name": index["name"], "sha256": index["sha256"]})
        return sorted(entries, key=lambda f: f["name"])

    def metadata_view(manifest):
        return [f["metadata"] for f in sorted(manifest["files"], key=lambda f: f["name"])]

    result["files"] = {"a": file_view(a), "b": file_view(b)}
    tensors_identical = not (result["changed"] or result["only_in_a"] or result["only_in_b"])
    files_identical = result["files"]["a"] == result["files"]["b"]
    if not tensors_identical:
        relation = "tensors_differ"
    elif files_identical:
        relation = "tensors_and_files_identical"
    else:
        relation = "tensors_identical_files_differ"
    result.update(
        scope=a["scope"],
        status="passed",
        tensors_identical=tensors_identical,
        files_identical=files_identical,
        metadata_identical=metadata_view(a) == metadata_view(b),
        relation=relation,
    )
    return result
