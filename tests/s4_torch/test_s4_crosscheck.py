"""Cross-check: gate.tensor_manifest and teacher_rl.params give identical rule-11 tensor hashes.

Build-spec B3 item (9). ``tests/s4_torch/test_s4_params.py`` already checks the literal
``tiny.safetensors`` (F32 + U8 scalar) and a toy ``state_dict`` with BF16/I64 entries. This file
extends coverage without duplicating it:

- every one of the twelve rule-11 dtypes, in both directions (tensor -> saved bytes -> torch-free
  manifest, and saved bytes -> ``safetensors`` load -> ``params``);
- an independent writer: hand-packed safetensors bytes (``struct``) whose per-tensor oracle is
  ``hashlib.sha256`` of the literal byte range, never either module;
- non-contiguous views (transpose, strided slice) and zero-element tensors;
- a sharded checkpoint with an index, read by ``build_manifest_from_directory``, whose
  ``tensor_set_sha256`` equals the ``params.snapshot`` of the same state;
- E07 through ``teacher_rl.adapters`` on a real AdamW: a skipped zero-gradient minibatch passes
  ``check_skip_semantics``; stepping on a zero gradient is caught.
"""

import hashlib
import json
import struct

import pytest
import torch
from safetensors.torch import load_file, save_file

from gate.tensor_manifest import build_manifest, build_manifest_from_directory
from teacher_rl import adapters, params

KEYS = ("dtype", "shape", "byte_length", "sha256")
DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
    "U8": torch.uint8, "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


def every_dtype_state():
    state = {}
    for name, dtype in DTYPES.items():
        if dtype is torch.bool:
            value = torch.tensor([[True, False, True], [False, False, True]])
        elif dtype.is_floating_point:
            value = torch.tensor([[1.0, -2.0, 0.5], [3.25, -0.0, 4.0]]).to(dtype)
        else:
            value = torch.tensor([[1, -2, 3], [4, 5, -6]]).to(dtype)
        state[f"t.{name.lower()}"] = value
        state[f"scalar.{name.lower()}"] = value.reshape(-1)[0].clone()
    state["empty.f32"] = torch.zeros((0, 3), dtype=torch.float32)
    return state


def assert_same(manifest, identities):
    assert manifest["status"] == "passed", manifest["issues"]
    assert set(manifest["tensors"]) == set(identities)
    for name, identity in identities.items():
        for key in KEYS:
            assert identity[key] == manifest["tensors"][name][key], (name, key)


def test_every_rule11_dtype_both_directions(tmp_path):
    state = every_dtype_state()
    path = tmp_path / "all.safetensors"
    save_file(state, str(path))
    manifest = build_manifest({"all.safetensors": path.read_bytes()}, scope="synthetic")
    assert {e["dtype"] for e in manifest["tensors"].values()} == set(DTYPES)
    assert_same(manifest, {n: params.tensor_identity(t) for n, t in state.items()})
    assert_same(manifest, {n: params.tensor_identity(t) for n, t in load_file(str(path)).items()})
    assert manifest["tensors"]["empty.f32"]["sha256"] == hashlib.sha256(b"").hexdigest()
    assert manifest["tensors"]["empty.f32"]["shape"] == [0, 3]
    snap = params.snapshot(state, scope="synthetic")
    assert snap["tensor_set_sha256"] == manifest["tensor_set_sha256"]


def _pack(tensors):
    """Independent safetensors writer: header offsets in insertion order, no padding."""
    header, blobs, offset = {}, [], 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset,
                                                                         offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    text = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<Q", len(text)) + text + b"".join(blobs)


HAND = {
    "h.f16": ("F16", [2], struct.pack("<2e", 1.0, -2.0)),
    "h.bf16": ("BF16", [2], bytes.fromhex("803f00c0")),  # 1.0, -2.0 (upper float32 halves)
    "h.i16": ("I16", [2], struct.pack("<2h", -1, 2)),
    "h.bool": ("BOOL", [2], b"\x01\x00"),
    "h.f64": ("F64", [], struct.pack("<d", 0.1)),
    "h.f8e4m3": ("F8_E4M3", [2], bytes([0x38, 0xC0])),  # 1.0, -2.0 in E4M3
}


def test_hand_written_bytes_load_and_hash_to_the_literal_byte_ranges(tmp_path):
    path = tmp_path / "hand.safetensors"
    path.write_bytes(_pack(HAND))
    oracle = {name: hashlib.sha256(raw).hexdigest() for name, (_, _, raw) in HAND.items()}
    manifest = build_manifest({"hand.safetensors": path.read_bytes()}, scope="synthetic")
    assert {n: e["sha256"] for n, e in manifest["tensors"].items()} == oracle
    loaded = load_file(str(path))
    assert loaded["h.bf16"].tolist() == [1.0, -2.0]
    assert loaded["h.f8e4m3"].float().tolist() == [1.0, -2.0]
    assert {n: params.tensor_identity(t)["sha256"] for n, t in loaded.items()} == oracle
    assert_same(manifest, {n: params.tensor_identity(t) for n, t in loaded.items()})


def test_non_contiguous_views_hash_their_saved_logical_bytes(tmp_path):
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    views = {"transposed": base.T, "strided": base[:, ::2], "row": base[1]}
    assert not views["transposed"].is_contiguous() and not views["strided"].is_contiguous()
    path = tmp_path / "views.safetensors"
    save_file({n: v.contiguous() for n, v in views.items()}, str(path))
    manifest = build_manifest({"views.safetensors": path.read_bytes()}, scope="synthetic")
    assert_same(manifest, {n: params.tensor_identity(v) for n, v in views.items()})
    # base.T in row-major order is base's columns: the logical values, never the storage
    column_major = struct.pack("<12f", 0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11)
    assert manifest["tensors"]["transposed"]["sha256"] == hashlib.sha256(column_major).hexdigest()


def test_sharded_checkpoint_tensor_set_equals_the_params_snapshot(tmp_path):
    state = every_dtype_state()
    names = sorted(state)
    shards = {"model-00001-of-00002.safetensors": names[: len(names) // 2],
              "model-00002-of-00002.safetensors": names[len(names) // 2:]}
    weight_map = {}
    for shard, members in shards.items():
        save_file({n: state[n] for n in members}, str(tmp_path / shard))
        weight_map.update({n: shard for n in members})
    index = "model.safetensors.index.json"
    (tmp_path / index).write_text(json.dumps({"weight_map": weight_map}))
    manifest = build_manifest_from_directory(tmp_path, None, scope="synthetic",
                                             index_name=index)
    assert manifest["layout"] == "sharded" and manifest["totals"]["file_count"] == 2
    assert_same(manifest, {n: params.tensor_identity(t) for n, t in state.items()})
    assert manifest["tensor_set_sha256"] == params.snapshot(state, scope="synthetic")[
        "tensor_set_sha256"]


def test_a_changed_byte_is_seen_by_both_modules(tmp_path):
    state = {"w": torch.tensor([1.0, 2.0])}
    path = tmp_path / "w.safetensors"
    save_file(state, str(path))
    before = build_manifest({"w.safetensors": path.read_bytes()}, scope="synthetic")
    state["w"][1] = torch.nextafter(torch.tensor(2.0), torch.tensor(3.0))  # one ulp
    save_file(state, str(path))
    after = build_manifest({"w.safetensors": path.read_bytes()}, scope="synthetic")
    assert before["tensors"]["w"]["sha256"] != after["tensors"]["w"]["sha256"]
    assert params.tensor_identity(state["w"])["sha256"] == after["tensors"]["w"]["sha256"]


def _toy():
    torch.manual_seed(0)
    model = torch.nn.Module()
    model.expert = torch.nn.Linear(2, 2)
    model.vlm = torch.nn.Linear(2, 1)
    model.vlm.requires_grad_(False)
    return model


def _snap(model, optimizer):
    return (params.snapshot(model, scope="synthetic"),
            params.optimizer_state_snapshot(optimizer, model, scope="synthetic"))


def _sched(scheduler):
    return adapters.scheduler_state(scheduler)


def test_e07_adapter_over_real_adamw_skip_then_step():
    model = _toy()
    optimizer = torch.optim.AdamW(model.expert.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    before_sched = _sched(scheduler)
    s0 = _snap(model, optimizer)
    assert adapters.skip_snapshot(*s0)["step_counters"] == {"expert.bias": 0,
                                                            "expert.weight": 0}
    # zero reduced gradient: upstream skips optimizer.step (trainer.py:541-547)
    optimizer.zero_grad()
    for p in model.expert.parameters():
        p.grad = torch.zeros_like(p)
    s1 = _snap(model, optimizer)
    # a real gradient: optimizer.step runs and advances every counter by one
    model.expert(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    s2 = _snap(model, optimizer)
    scheduler.step()
    rows = [adapters.minibatch_record(0, grad_norm=0.0, optimizer_step_called=False,
                                      before=s0, after=s1),
            adapters.minibatch_record(1, grad_norm=1.0, optimizer_step_called=True,
                                      before=s1, after=s2)]
    assert adapters.skip_snapshot(*s2)["step_counters"] == {"expert.bias": 1,
                                                            "expert.weight": 1}
    report = adapters.check_skip(0, scheduler_before=before_sched,
                                 scheduler_after=_sched(scheduler), minibatches=rows,
                                 scope="synthetic")
    assert report["status"] == "passed", report["issues"]
    assert "vlm.weight" in adapters.skip_snapshot(*s2)["parameters"]


def test_e07_adapter_catches_a_step_on_a_zero_gradient():
    model = _toy()
    optimizer = torch.optim.AdamW(model.expert.parameters(), lr=0.01, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    before_sched = _sched(scheduler)
    s0 = _snap(model, optimizer)
    for p in model.expert.parameters():
        p.grad = torch.zeros_like(p)
    optimizer.step()  # the defect E07 must catch: stepping although the norm is exactly 0
    s1 = _snap(model, optimizer)
    scheduler.step()
    rows = [adapters.minibatch_record(0, grad_norm=0.0, optimizer_step_called=True,
                                      before=s0, after=s1)]
    report = adapters.check_skip(0, scheduler_before=before_sched,
                                 scheduler_after=_sched(scheduler), minibatches=rows,
                                 scope="synthetic")
    codes = {i["code"] for i in report["issues"]}
    assert report["status"] == "blocked"
    assert {"skip_rule_violated", "step_counter_changed_at_skip",
            "optimizer_state_changed_at_skip", "parameters_changed_at_skip"} <= codes


@pytest.mark.parametrize("dtype", [torch.complex64, torch.uint16])
def test_dtypes_outside_rule11_are_refused_by_params(dtype):
    with pytest.raises(params.SnapshotError):
        params.tensor_identity(torch.zeros(2, dtype=dtype))
