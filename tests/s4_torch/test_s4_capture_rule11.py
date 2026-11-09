"""Proposed shared change (review F5-07): the capture's rule-11 byte path on real torch.

``teacher_rl.capture.entrypoint.tensor_entry`` / ``rule11_sha256`` must equal
``teacher_rl.params.tensor_identity`` and ``gate.tensor_manifest`` over the safetensors bytes
of the same tensor, for every rule-11 dtype and for views whose storage is not the logical
row-major order (transpose, strided slice, negated view). Oracle: sha256 of the byte range
``safetensors.torch.save`` writes.
"""

import hashlib
import json
import struct

import pytest
import torch
from safetensors.torch import save

from gate.tensor_manifest import build_manifest
from teacher_rl import params
from teacher_rl.capture import entrypoint

DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
    "U8": torch.uint8, "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


def logical(tensor):
    # safetensors.save writes a neg-bit view's raw storage; the logical values are the identity
    return tensor.resolve_neg().resolve_conj().contiguous()


def stored_sha(tensor):
    """sha256 of the exact byte range safetensors stores for ``t`` (independent oracle)."""
    blob = save({"t": logical(tensor)})
    (size,) = struct.unpack("<Q", blob[:8])
    begin, end = json.loads(blob[8:8 + size])["t"]["data_offsets"]
    return hashlib.sha256(blob[8 + size + begin:8 + size + end]).hexdigest()


def cases():
    out = {}
    for name, dtype in DTYPES.items():
        base = torch.arange(-6, 6, dtype=torch.float32).reshape(3, 4)
        out[f"{name}/plain"] = base.to(dtype)
        out[f"{name}/transpose"] = base.to(dtype).t()
        out[f"{name}/strided"] = base.to(dtype)[:, ::2]
    out["F32/neg_view"] = torch._neg_view(torch.arange(4, dtype=torch.float32))
    out["F32/empty"] = torch.zeros(0, 3)
    return out


@pytest.mark.parametrize("label", sorted(cases()))
def test_capture_hash_equals_params_and_tensor_manifest(label):
    tensor = cases()[label]
    entry = entrypoint.tensor_entry(tensor, "param", "sha256_local")
    oracle = stored_sha(tensor)
    assert entry["sha256"] == oracle
    assert entry["dtype"] == label.split("/")[0] and entry["hash_scope"] == "full"
    assert params.tensor_identity(tensor)["sha256"] == oracle
    blob = save({"t": logical(tensor)})
    manifest = build_manifest({"t.safetensors": blob}, scope="synthetic")
    assert manifest["tensors"]["t"]["sha256"] == oracle
