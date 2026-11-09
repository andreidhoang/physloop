"""Known answers for S4 parameter/buffer snapshots, diffs and E03/E07 torch checks (A10).

Oracles are independent of ``teacher_rl.params``: rule-11 hashes are SHA-256 of bytes built
with ``struct`` by hand; the saved-byte-range check uses ``gate.tensor_manifest`` (torch-free
safetensors parser) over files written by ``safetensors.torch.save_file``; changed-fraction
literals are counted from the toy model's declared shapes. The toy mirrors the pinned
Alpamayo module tree (``vlm.model.{visual,language_model}``, ``vlm.lm_head``,
``expert.{layers,norm}``, ``action_in_proj`` with a ``freqs`` buffer, ``action_out_proj``,
``action_space`` mean/std buffers) without tying ``lm_head`` (safetensors refuses shared
storage and ``named_parameters`` would drop one name).
"""

import hashlib
import json
import struct

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from gate.s2_identity import identity_hash
from gate.tensor_manifest import build_manifest
from scripts import s4_snapshot_params
from teacher_rl import export_audit, params, parity

TRAINABLE = ("expert.", "action_in_proj.", "action_out_proj.")
FROZEN = ("vlm.",)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = nn.Module()
        self.vlm.model = nn.Module()
        self.vlm.model.visual = nn.Linear(4, 4)
        self.vlm.model.language_model = nn.Module()
        self.vlm.model.language_model.layers = nn.ModuleList([nn.Linear(4, 4)])
        self.vlm.lm_head = nn.Linear(4, 6, bias=False)
        self.expert = nn.Module()
        self.expert.layers = nn.ModuleList([nn.Linear(4, 4)])
        self.expert.norm = nn.LayerNorm(4)
        self.action_in_proj = nn.Module()
        self.action_in_proj.proj = nn.Linear(4, 4)
        self.action_in_proj.register_buffer("freqs", torch.tensor([[1.0, 10.0]]))
        self.action_out_proj = nn.Linear(4, 2)
        self.action_space = nn.Module()
        self.action_space.register_buffer("accel_mean", torch.tensor(0.25))
        self.action_space.register_buffer("accel_std", torch.tensor(2.0))
        self.action_space.register_buffer("scratch", torch.zeros(1), persistent=False)

    def forward(self, x, a):
        h = self.vlm.model.language_model.layers[0](torch.tanh(self.vlm.model.visual(x)))
        logits = self.vlm.lm_head(h)
        z = (a - self.action_space.accel_mean) / self.action_space.accel_std
        enc = torch.cat([torch.sin(z * self.action_in_proj.freqs),
                         torch.cos(z * self.action_in_proj.freqs)], dim=-1)
        e = self.expert.norm(self.expert.layers[0](h + self.action_in_proj.proj(enc)))
        return logits, self.action_out_proj(e)


def _toy(freeze_vlm=True):
    torch.manual_seed(0)
    model = Toy()
    if freeze_vlm:
        for p in model.vlm.parameters():
            p.requires_grad_(False)
    return model


def _loss(model):
    torch.manual_seed(1)
    x, a = torch.randn(3, 4), torch.randn(3, 1)
    logits, out = model(x, a)
    return out.pow(2).sum() + logits.pow(2).sum()


# ---------------------------------------------------------------------------
# Rule-11 tensor identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tensor", "dtype", "shape", "raw"),
    [
        (torch.tensor([1.0, 2.0]), "F32", [2], struct.pack("<2f", 1.0, 2.0)),
        (torch.tensor([1.0], dtype=torch.bfloat16), "BF16", [1], b"\x80\x3f"),
        (torch.tensor([1.0, -2.0], dtype=torch.float16), "F16", [2], struct.pack("<2e", 1, -2)),
        (torch.tensor(7), "I64", [], struct.pack("<q", 7)),
        (torch.tensor([3], dtype=torch.int32), "I32", [1], struct.pack("<i", 3)),
        (torch.tensor([True, False]), "BOOL", [2], b"\x01\x00"),
        (torch.tensor([255], dtype=torch.uint8), "U8", [1], b"\xff"),
        (torch.tensor([1.0]).to(torch.float8_e4m3fn), "F8_E4M3", [1], b"\x38"),
        (torch.tensor([1.0]).to(torch.float8_e5m2), "F8_E5M2", [1], b"\x3c"),
        (torch.tensor([0.5], dtype=torch.float64), "F64", [1], struct.pack("<d", 0.5)),
        (torch.empty(0, 3), "F32", [0, 3], b""),
    ],
)
def test_tensor_identity_literals(tensor, dtype, shape, raw):
    ident = params.tensor_identity(tensor)
    assert ident == {
        "dtype": dtype,
        "shape": shape,
        "numel": tensor.numel(),
        "byte_length": len(raw),
        "sha256": _sha(raw),
    }


def test_non_contiguous_view_hashes_logical_row_major_order():
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    view = base.T
    assert not view.is_contiguous()
    expected = _sha(struct.pack("<6f", 0, 3, 1, 4, 2, 5))
    assert params.tensor_identity(view)["sha256"] == expected
    assert params.tensor_identity(view.contiguous())["sha256"] == expected
    assert expected != _sha(struct.pack("<6f", 0, 1, 2, 3, 4, 5))  # storage order differs
    neg = torch.tensor([1.0, 2.0])._neg_view()
    assert params.tensor_identity(neg)["sha256"] == _sha(struct.pack("<2f", -1.0, -2.0))


@pytest.mark.parametrize(
    "tensor",
    [
        torch.tensor([1 + 1j], dtype=torch.complex64),
        torch.tensor([1], dtype=torch.uint16),
        torch.tensor([1], dtype=torch.uint32),
        torch.tensor([1.0]).to(torch.float8_e4m3fnuz),
        torch.zeros(2, device="meta"),
        torch.sparse_coo_tensor([[0]], [1.0], (2,)),
    ],
)
def test_tensor_identity_rejects_unsupported(tensor):
    with pytest.raises(params.SnapshotError):
        params.tensor_identity(tensor)


def test_tensor_identity_rejects_subclasses_and_non_tensors():
    class Sub(torch.Tensor):
        pass

    with pytest.raises(params.SnapshotError, match="subclass"):
        params.tensor_identity(torch.tensor([1.0]).as_subclass(Sub))
    with pytest.raises(params.SnapshotError):
        params.tensor_identity([1.0])
    assert params.tensor_identity(nn.Parameter(torch.tensor([1.0])))["sha256"] == _sha(
        struct.pack("<f", 1.0)
    )


def test_loaded_hash_equals_saved_safetensors_byte_range(tmp_path):
    model = _toy()
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    state["half"] = torch.tensor([1.5, -2.25], dtype=torch.bfloat16)
    state["scalar"] = torch.tensor(3, dtype=torch.int64)
    path = tmp_path / "model.safetensors"
    save_file(state, str(path))
    manifest = build_manifest({"model.safetensors": path.read_bytes()}, scope="synthetic")
    assert manifest["status"] == "passed"
    snap = params.snapshot(state, scope="synthetic")
    loaded = params.snapshot(load_file(str(path)), scope="synthetic")
    assert set(manifest["tensors"]) == set(snap["tensors"]) == set(loaded["tensors"])
    for name, entry in manifest["tensors"].items():
        for key in ("dtype", "shape", "byte_length", "sha256"):
            assert snap["tensors"][name][key] == entry[key] == loaded["tensors"][name][key]
    assert snap["tensor_set_sha256"] == manifest["tensor_set_sha256"]


def test_tensor_identity_uses_the_shared_rule11_helpers(monkeypatch):
    # A10R: one rule-11 implementation; params delegates to gate.tensor_manifest.
    from gate import tensor_manifest

    calls = []

    def spy(dtype, shape, raw):
        calls.append((dtype, list(shape)))
        return tensor_manifest.tensor_record(dtype, shape, raw)

    monkeypatch.setattr(params, "tensor_record", spy)
    identity = params.tensor_identity(torch.tensor([1.0, -2.0]))
    assert calls == [("F32", [2])]
    assert identity["sha256"] == tensor_manifest.tensor_sha256(struct.pack("<2f", 1.0, -2.0))
    assert params.HASH_CONVENTION == tensor_manifest.HASH_CONVENTION
    assert set(params._DTYPE_NAMES.values()) == set(tensor_manifest.DTYPE_ITEMSIZE)


# The literal ``tiny.safetensors`` of tests/test_tensor_manifest.py (A6): tensor "a" F32
# [1.0, -2.0], tensor "b" U8 scalar 7; hashes computed there with ``shasum -a 256``.
_TINY_HEADER = (
    b'{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},'
    b'"b":{"dtype":"U8","shape":[],"data_offsets":[8,9]}}'
)
_TINY = struct.pack("<Q", 105) + _TINY_HEADER + bytes.fromhex("0000803f000000c0") + b"\x07"
_TINY_SHA = {"a": "ee4ac73c2bd27756ab82780f27c73a7bc4d3f0bb6acb37e008bc27eccd7e588b",
             "b": "ca358758f6d27e6cf45272937977a748fd88391db679ceda7dc7bf1f005ee879"}


def test_params_hashes_equal_tensor_manifest_on_tiny_safetensors(tmp_path):
    # A10R / B3 cross-check: safetensors-loaded tensors hashed by teacher_rl.params equal
    # gate.tensor_manifest's manifest of the same file, tensor by tensor.
    path = tmp_path / "tiny.safetensors"
    path.write_bytes(_TINY)
    manifest = build_manifest({"tiny.safetensors": path.read_bytes()}, scope="synthetic")
    assert manifest["status"] == "passed"
    loaded = load_file(str(path))
    assert set(loaded) == set(manifest["tensors"]) == set(_TINY_SHA)
    for name, tensor in loaded.items():
        identity = params.tensor_identity(tensor)
        entry = manifest["tensors"][name]
        for key in ("dtype", "shape", "byte_length", "sha256"):
            assert identity[key] == entry[key], (name, key)
        assert identity["sha256"] == _TINY_SHA[name]


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------


def test_snapshot_module_known_answer():
    model = _toy()
    model.vlm.eval()
    snap = params.snapshot(model, scope="synthetic")
    assert snap["artifact_kind"] == "s4_param_snapshot" and snap["schema_version"] == 1
    assert snap["status"] == "passed" and snap["source"] == "module" and snap["issues"] == []
    tensors = snap["tensors"]
    assert list(tensors) == sorted(tensors)
    kinds = {n: e["kind"] for n, e in tensors.items()}
    assert sum(k == "parameter" for k in kinds.values()) == 13
    assert sorted(n for n, k in kinds.items() if k == "buffer") == [
        "action_in_proj.freqs",
        "action_space.accel_mean",
        "action_space.accel_std",
        "action_space.scratch",
    ]
    assert tensors["action_space.scratch"]["persistent"] is False
    assert tensors["action_space.accel_mean"]["persistent"] is True
    assert tensors["expert.norm.weight"]["persistent"] is None
    assert tensors["vlm.lm_head.weight"]["requires_grad"] is False
    assert tensors["vlm.lm_head.weight"]["shape"] == [6, 4]
    assert tensors["expert.layers.0.weight"]["requires_grad"] is True
    assert tensors["action_space.accel_mean"] == {
        "kind": "buffer", "dtype": "F32", "shape": [], "numel": 1, "byte_length": 4,
        "sha256": _sha(struct.pack("<f", 0.25)), "requires_grad": False, "persistent": True,
    }
    assert snap["modules"][""] is True and snap["modules"]["vlm"] is False
    assert snap["modules"]["vlm.model.visual"] is False and snap["modules"]["expert"] is True
    assert snap["totals"] == {"tensor_count": 17, "parameter_count": 13, "buffer_count": 4,
                              "numel": 64 + 28 + 20 + 10 + 2 + 3, "byte_length": 4 * 127}
    assert params.snapshot(model, scope="synthetic") == snap  # deterministic
    json.dumps(snap, allow_nan=False)


def test_snapshot_without_buffers_and_from_state_dict():
    model = _toy()
    snap = params.snapshot(model, scope="synthetic", buffers=False)
    assert all(e["kind"] == "parameter" for e in snap["tensors"].values())
    state = params.snapshot(model.state_dict(), scope="synthetic")
    assert state["source"] == "state_dict" and state["modules"] is None
    entry = state["tensors"]["expert.norm.weight"]
    assert entry["kind"] == "state" and entry["requires_grad"] is None
    assert entry["persistent"] is None
    assert "action_space.scratch" not in state["tensors"]  # non-persistent: not in state_dict


def test_snapshot_rejects_bad_inputs():
    with pytest.raises(ValueError, match="scope"):
        params.snapshot(_toy(), scope="real")
    with pytest.raises(params.SnapshotError):
        params.snapshot({"a": torch.tensor([1j])}, scope="synthetic")
    with pytest.raises(ValueError, match="names"):
        params.snapshot({"": torch.tensor([1.0])}, scope="synthetic")
    with pytest.raises(TypeError):
        params.snapshot([torch.tensor([1.0])], scope="synthetic")
    with pytest.raises(ValueError, match="empty"):
        params.snapshot({}, scope="synthetic")


# ---------------------------------------------------------------------------
# E03: one AdamW step, trainable set, diff by prefix
# ---------------------------------------------------------------------------


def _step(model, parameters):
    optimizer = torch.optim.AdamW(parameters, lr=0.1, weight_decay=0.01)
    optimizer.zero_grad()
    _loss(model).backward()
    optimizer.step()
    return optimizer


def test_e03_one_step_changes_only_declared_trainable_tensors():
    model = _toy()
    before = params.snapshot(model, scope="synthetic")
    _step(model, [p for p in model.parameters() if p.requires_grad])
    after = params.snapshot(model, scope="synthetic")
    d = params.diff(before, after)
    assert d["status"] == "passed"
    assert d["only_in_before"] == d["only_in_after"] == []
    assert d["dtype_changed"] == d["shape_changed"] == d["requires_grad_changed"] == []
    assert d["changed"] == [
        "action_in_proj.proj.bias", "action_in_proj.proj.weight",
        "action_out_proj.bias", "action_out_proj.weight",
        "expert.layers.0.bias", "expert.layers.0.weight",
        "expert.norm.bias", "expert.norm.weight",
    ]
    assert d["by_prefix"] == {
        "vlm.": {"tensors": 5, "changed_tensors": 0, "changed_tensor_fraction": 0.0,
                 "numel": 64, "changed_numel": 0, "changed_numel_fraction": 0.0,
                 "uncompared_tensors": 0},
        "expert.": {"tensors": 4, "changed_tensors": 4, "changed_tensor_fraction": 1.0,
                    "numel": 28, "changed_numel": 28, "changed_numel_fraction": 1.0,
                    "uncompared_tensors": 0},
        "action_in_proj.": {"tensors": 3, "changed_tensors": 2,
                            "changed_tensor_fraction": 2 / 3, "numel": 22,
                            "changed_numel": 20, "changed_numel_fraction": 20 / 22,
                            "uncompared_tensors": 0},
        "action_out_proj.": {"tensors": 2, "changed_tensors": 2,
                             "changed_tensor_fraction": 1.0, "numel": 10,
                             "changed_numel": 10, "changed_numel_fraction": 1.0,
                             "uncompared_tensors": 0},
        "(other)": {"tensors": 3, "changed_tensors": 0, "changed_tensor_fraction": 0.0,
                    "numel": 3, "changed_numel": 0, "changed_numel_fraction": 0.0,
                    "uncompared_tensors": 0},
    }
    check = params.check_trainable_set(before, TRAINABLE, FROZEN, diff=d)
    assert check["status"] == "passed", check["issues"]
    assert check["trainable_dtypes"] == {"F32": 8} and check["frozen_dtypes"] == {"F32": 5}
    assert check["explicit"]["vlm.lm_head.weight"] == {
        "present": True, "classification": "frozen", "requires_grad": False, "changed": False,
    }
    assert check["changed_frozen"] == check["changed_buffers"] == check["unclassified"] == []
    assert check["unchanged_trainable"] == []
    assert len(check["buffers"]) == 4


def test_e03_expert_only_optimizer_leaves_other_trainables_unchanged():
    model = _toy()
    before = params.snapshot(model, scope="synthetic")
    _step(model, model.expert.parameters())
    d = params.diff(before, params.snapshot(model, scope="synthetic"))
    assert d["by_prefix"]["expert."]["changed_tensors"] == 4
    assert d["by_prefix"]["action_in_proj."]["changed_tensors"] == 0
    check = params.check_trainable_set(before, TRAINABLE, FROZEN, diff=d)
    assert check["status"] == "passed"
    assert check["unchanged_trainable"] == [
        "action_in_proj.proj.bias", "action_in_proj.proj.weight",
        "action_out_proj.bias", "action_out_proj.weight",
    ]
    assert {i["code"] for i in check["issues"]} == {"trainable_unchanged"}
    assert {i["severity"] for i in check["issues"]} == {"info"}


def test_e03_negative_control_unfrozen_lm_head_is_caught():
    model = _toy()
    model.vlm.lm_head.weight.requires_grad_(True)
    before = params.snapshot(model, scope="synthetic")
    _step(model, [p for p in model.parameters() if p.requires_grad])
    d = params.diff(before, params.snapshot(model, scope="synthetic"))
    check = params.check_trainable_set(before, TRAINABLE, FROZEN, diff=d)
    assert check["status"] == "blocked"
    codes = {(i["code"], i["path"]) for i in check["issues"] if i["severity"] == "error"}
    assert codes == {
        ("frozen_parameter_requires_grad", "vlm.lm_head.weight"),
        ("frozen_parameter_changed", "vlm.lm_head.weight"),
    }
    assert check["explicit"]["vlm.lm_head.weight"]["changed"] is True


def test_trainable_set_negative_controls():
    model = _toy()
    model.expert.norm.bias.requires_grad_(False)
    snap = params.snapshot(model, scope="synthetic")
    check = params.check_trainable_set(snap, ("expert.",), FROZEN)
    codes = {(i["code"], i["path"]) for i in check["issues"] if i["severity"] == "error"}
    assert ("trainable_parameter_not_requires_grad", "expert.norm.bias") in codes
    assert ("tensor_unclassified", "action_out_proj.weight") in codes
    assert check["status"] == "blocked"

    overlap = params.check_trainable_set(snap, ("expert.",), ("expert.norm.",))
    assert "prefix_overlap" in {i["code"] for i in overlap["issues"]}

    state = params.snapshot(model.state_dict(), scope="synthetic")
    unknown = params.check_trainable_set(state, TRAINABLE, FROZEN)
    assert unknown["status"] == "blocked"
    assert "requires_grad_unknown" in {i["code"] for i in unknown["issues"]}

    with pytest.raises(ValueError, match="prefixes"):
        params.check_trainable_set(snap, (), FROZEN)
    with pytest.raises(ValueError, match="prefixes"):
        params.check_trainable_set(snap, "expert.", FROZEN)


def test_trainable_set_records_bf16_trainables_and_buffer_change():
    model = _toy()
    model.expert.to(torch.bfloat16)
    before = params.snapshot(model, scope="synthetic")
    model.action_space.accel_mean.add_(1.0)
    d = params.diff(before, params.snapshot(model, scope="synthetic"))
    check = params.check_trainable_set(before, TRAINABLE, FROZEN, diff=d)
    assert check["trainable_dtypes"] == {"BF16": 4, "F32": 4}
    codes = {i["code"]: i["severity"] for i in check["issues"]}
    assert codes["trainable_not_float32"] == "warning"
    assert codes["buffer_changed"] == "error"
    assert check["changed_buffers"] == ["action_space.accel_mean"]


def test_trainable_set_rejects_diff_for_other_snapshot():
    a = params.snapshot(_toy(), scope="synthetic")
    other = _toy()
    other.expert.norm.weight.data.add_(1.0)
    b = params.snapshot(other, scope="synthetic")
    d = params.diff(b, b)
    check = params.check_trainable_set(a, TRAINABLE, FROZEN, diff=d)
    assert "diff_not_for_manifest" in {i["code"] for i in check["issues"]}


def test_diff_reports_keys_dtypes_shapes_and_training_flags():
    model = _toy()
    before = params.snapshot(model, scope="synthetic")
    model.vlm.eval()
    model.expert.norm = nn.LayerNorm(4, dtype=torch.float64)
    model.action_out_proj = nn.Linear(4, 3)
    model.extra = nn.Linear(1, 1, bias=False)
    del model.action_in_proj.freqs
    after = params.snapshot(model, scope="synthetic")
    d = params.diff(before, after)
    assert d["only_in_before"] == ["action_in_proj.freqs"]
    assert d["only_in_after"] == ["extra.weight"]
    assert d["dtype_changed"] == ["expert.norm.bias", "expert.norm.weight"]
    assert d["shape_changed"] == ["action_out_proj.bias", "action_out_proj.weight"]
    assert d["training_changed"] == [
        "vlm", "vlm.lm_head", "vlm.model", "vlm.model.language_model",
        "vlm.model.language_model.layers", "vlm.model.language_model.layers.0",
        "vlm.model.visual",
    ]
    assert d["modules_only_in_after"] == ["extra"]


def test_diff_fails_closed_on_tampered_or_mismatched_inputs():
    snap = params.snapshot(_toy(), scope="synthetic")
    tampered = json.loads(json.dumps(snap))
    tampered["tensors"]["expert.norm.weight"]["sha256"] = "0" * 64
    d = params.diff(snap, tampered)
    assert d["status"] == "blocked" and d["changed"] is None
    assert [i["code"] for i in d["issues"]] == ["input_invalid"]
    other = params.snapshot(_toy(), scope="actual")
    assert [i["code"] for i in params.diff(snap, other)["issues"]] == ["scope_mismatch"]


def test_diff_empty_prefix_bucket_is_not_estimable():
    snap = params.snapshot({"a.x": torch.tensor([1.0])}, scope="synthetic")
    d = params.diff(snap, snap, prefixes=("a.", "b."))
    assert d["by_prefix"]["b."] == {"tensors": 0, "changed_tensors": 0,
                                    "changed_tensor_fraction": None, "numel": 0,
                                    "changed_numel": 0, "changed_numel_fraction": None,
                                    "uncompared_tensors": 0}
    assert d["by_prefix"]["(other)"]["changed_tensor_fraction"] is None


def test_diff_groups_policy_wrapper_names_by_release_prefix():
    snap = params.snapshot(
        {"expert_model.expert.norm.weight": torch.tensor([1.0]),
         "expert_model.vlm.lm_head.weight": torch.tensor([2.0])},
        scope="synthetic",
    )
    d = params.diff(snap, snap)
    assert d["by_prefix"]["expert."]["tensors"] == 1
    assert d["by_prefix"]["vlm."]["tensors"] == 1


# ---------------------------------------------------------------------------
# E07 torch part: optimizer skip at exactly-zero reduced gradient norm
# ---------------------------------------------------------------------------


def _e07_setup():
    model = _toy()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=0.1, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (step + 1))
    optimizer.zero_grad()
    _loss(model).backward()
    optimizer.step()  # one real step so optimizer moments exist
    scheduler.step()
    optimizer.zero_grad()
    (_loss(model) * 0.0).backward()  # every reduced gradient is exactly zero
    norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    assert float(norm) == 0.0
    return model, optimizer, scheduler


def test_e07_skipped_step_leaves_parameters_and_moments_unchanged():
    model, optimizer, scheduler = _e07_setup()
    p0 = params.snapshot(model, scope="synthetic")
    o0 = params.optimizer_state_snapshot(optimizer, model, scope="synthetic")
    grad_norm_value = 0.0
    if grad_norm_value != 0.0:  # pinned AlpaGym trainer.py:541-547 gate
        optimizer.step()
    scheduler.step()  # the pinned trainer always advances the scheduler
    p1 = params.snapshot(model, scope="synthetic")
    o1 = params.optimizer_state_snapshot(optimizer, model, scope="synthetic")
    assert params.diff(p0, p1)["changed"] == []
    cmp = params.compare_optimizer_state(o0, o1)
    assert cmp["status"] == "passed"
    assert cmp["state_identical"] is True and cmp["changed_state"] == []
    assert cmp["param_membership_identical"] is True
    assert cmp["hyperparameters_changed"] == [
        {"group": 0, "key": "lr", "before": 0.05, "after": 0.1 / 3},
    ]
    entry = o0["state"]["expert.norm.weight"]
    assert entry["step"]["value"] == 1.0
    assert set(entry) == {"exp_avg", "exp_avg_sq", "step"}
    assert entry["exp_avg"]["dtype"] == "F32" and entry["exp_avg"]["shape"] == [4]


def test_e07_negative_control_stepping_on_zero_gradient_changes_state():
    model, optimizer, _ = _e07_setup()
    p0 = params.snapshot(model, scope="synthetic")
    o0 = params.optimizer_state_snapshot(optimizer, model, scope="synthetic")
    optimizer.step()  # what the upstream gate prevents: decay and moment decay still apply
    d = params.diff(p0, params.snapshot(model, scope="synthetic"))
    assert d["by_prefix"]["expert."]["changed_tensors"] == 4
    cmp = params.compare_optimizer_state(
        o0, params.optimizer_state_snapshot(optimizer, model, scope="synthetic")
    )
    assert cmp["state_identical"] is False
    assert "expert.norm.weight" in cmp["changed_state"]
    o1 = params.optimizer_state_snapshot(optimizer, model, scope="synthetic")
    assert o1["state"]["expert.norm.weight"]["step"]["value"] == 2.0


def test_optimizer_snapshot_requires_named_parameters():
    model = _toy()
    stray = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([*model.expert.parameters(), stray], lr=0.1)
    with pytest.raises(params.SnapshotError, match="name"):
        params.optimizer_state_snapshot(optimizer, model, scope="synthetic")
    snap = params.optimizer_state_snapshot(
        torch.optim.AdamW(model.expert.parameters(), lr=0.1), model, scope="synthetic"
    )
    assert snap["state"] == {}  # no step taken yet: no moments
    assert snap["param_groups"][0]["params"] == [
        "expert.layers.0.weight", "expert.layers.0.bias", "expert.norm.weight",
        "expert.norm.bias",
    ]
    assert snap["param_groups"][0]["hyperparameters"]["betas"] == [0.9, 0.999]
    assert snap["optimizer_class"] == "AdamW"


def test_compare_optimizer_state_rejects_invalid_input():
    model = _toy()
    snap = params.optimizer_state_snapshot(
        torch.optim.AdamW(model.expert.parameters(), lr=0.1), model, scope="synthetic"
    )
    broken = dict(snap, schema_version=2)
    assert params.compare_optimizer_state(snap, broken)["status"] == "blocked"


# ---------------------------------------------------------------------------
# Toy E02: Cosmos-style zero-update export audited end to end (L1)
# ---------------------------------------------------------------------------


def test_toy_zero_update_export_meets_e02_engineering_requirements(tmp_path):
    # A10-F2: every parameter is cast, so no same-dtype tensor is compared
    # (not_applicable, never passed) and E02 stays qualify_on_host, never passed.
    # A10-F7: the Gym bundle is a real symlink bundle recorded by the links CLI.
    tmp_path = tmp_path.resolve()
    model = _toy()
    release = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(release, str(tmp_path / "release.safetensors"))
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "release.safetensors").symlink_to(tmp_path / "release.safetensors")
    # Mimic the pinned export: named_parameters only, mapped names, bf16 cast, 1 shard.
    exported = {
        export_audit.policy_name_to_export_key(f"expert_model.{n}", hf_config_is_vlm=True):
        p.detach().to(torch.bfloat16).contiguous()
        for n, p in model.named_parameters()
    }
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    save_file(exported, str(export_dir / "00000.safetensors"))
    index = {"metadata": {"total_size": sum(t.numel() * 2 for t in exported.values())},
             "weight_map": {k: "00000.safetensors" for k in exported}}
    (export_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    inference_dir = tmp_path / "inference"
    inference_dir.mkdir()
    save_file({export_audit.export_key_to_inference_key(k): v for k, v in exported.items()},
              str(inference_dir / "00000.safetensors"), metadata={"format": "pt"})
    index["weight_map"] = {export_audit.export_key_to_inference_key(k): v
                           for k, v in index["weight_map"].items()}
    (inference_dir / "model.safetensors.index.json").write_text(json.dumps(index))

    def manifest(directory, *extra):
        argv = ["manifest", "--root", str(directory), "--scope", "synthetic", *extra]
        return s4_snapshot_params.build(argv)

    release_m = manifest(tmp_path, "--file", "release.safetensors")
    export_m = manifest(export_dir, "--index", "model.safetensors.index.json")
    inference_m = manifest(inference_dir, "--index", "model.safetensors.index.json")
    links = s4_snapshot_params.build([
        "links", "--bundle-root", str(bundle_dir), "--release-root", str(tmp_path),
        "--scope", "synthetic", "--file", "release.safetensors",
    ])
    assert links["status"] == "passed"
    policy = params.snapshot(model, scope="synthetic")
    report = export_audit.check_zero_update_export(
        release=release_m, export=export_m, inference=inference_m, policy_snapshot=policy,
        hf_config_is_vlm=True, optimizer_steps_applied=0, gym_bundle=links,
    )
    assert report["status"] == "qualify_on_host", report["issues"]
    assert [i for i in report["issues"] if i["severity"] == "error"] == []
    assert report["requirements"] == {
        "optimizer_steps_applied_zero": "passed",
        "gym_bundle_matches_release": "passed",
        "export_audit": "passed",
        "same_dtype_tensors_bitwise_identical": "not_applicable",
        "inference_conversion": "passed",
        "cast_tensor_parity": "qualify_on_host",
        "action_level_parity": "qualify_on_host",
    }
    assert report["same_dtype_tensors_compared"] == 0
    assert report["buffers_absent_from_export"] == [
        "action_in_proj.freqs", "action_space.accel_mean", "action_space.accel_std",
    ]
    assert len(report["requires_parity_ladder"]) == 13  # every F32 parameter was cast
    assert identity_hash(report) == identity_hash(
        export_audit.check_zero_update_export(
            release=release_m, export=export_m, inference=inference_m, policy_snapshot=policy,
            hf_config_is_vlm=True, optimizer_steps_applied=0, gym_bundle=links,
        )
    )
    # The cast parity this report defers is reachable: the inference conversion equals the
    # release rounded to BF16 bit for bit (reference cast to the candidate dtype).
    names = report["requires_parity_ladder"]
    ladder = parity.compare_tensors(
        {n: release[n] for n in names},
        {n: t for n, t in load_file(str(inference_dir / "00000.safetensors")).items()
         if n in names},
        tolerance=None, scope="synthetic", cross_dtype="reference_cast_to_candidate",
    )
    assert ladder["verdict"] == "identical" and ladder["status"] == "passed"


# ---------------------------------------------------------------------------
# Host script: read-only manifest/diff CLI (exercised in process)
# ---------------------------------------------------------------------------


def test_snapshot_script_manifest_and_diff(tmp_path, capsys):
    a = {"expert.w": torch.tensor([1.0, 2.0]), "vlm.w": torch.tensor([3.0])}
    b = {"expert.w": torch.tensor([1.0, 2.5]), "vlm.w": torch.tensor([3.0])}
    for name, tensors in (("a", a), ("b", b)):
        (tmp_path / name).mkdir()
        save_file(tensors, str(tmp_path / name / "model.safetensors"))
    for name in ("a", "b"):
        code = s4_snapshot_params.main([
            "manifest", "--root", str(tmp_path / name), "--scope", "synthetic",
            "--file", "model.safetensors", "--output", str(tmp_path / f"{name}.json"),
        ])
        assert code == 0
    manifest = json.loads((tmp_path / "a.json").read_text())
    assert manifest["tensors"]["expert.w"]["sha256"] == _sha(struct.pack("<2f", 1.0, 2.0))
    code = s4_snapshot_params.main(["diff", str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["changed"] == ["expert.w"]
    assert out["by_prefix"]["expert."]["changed_tensors"] == 1
    # Refuses to overwrite and refuses to write inside the checkpoint root.
    assert s4_snapshot_params.main([
        "manifest", "--root", str(tmp_path / "a"), "--scope", "synthetic",
        "--file", "model.safetensors", "--output", str(tmp_path / "a.json"),
    ]) == 2
    assert s4_snapshot_params.main([
        "manifest", "--root", str(tmp_path / "a"), "--scope", "synthetic",
        "--file", "model.safetensors", "--output", str(tmp_path / "a" / "m.json"),
    ]) == 2
    # Symlinked shard is refused (no-follow read).
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "model.safetensors").symlink_to(tmp_path / "a" / "model.safetensors")
    assert s4_snapshot_params.main([
        "manifest", "--root", str(tmp_path / "c"), "--scope", "synthetic",
        "--file", "model.safetensors",
    ]) == 2
    err = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert err["status"] == "blocked" and err["issues"][0]["code"] == "manifest_read_failed"


# ---------------------------------------------------------------------------
# Review regressions (A10-F1, A10-F6, A10-F7): key-set drift, CLI refusals, bundle links
# ---------------------------------------------------------------------------


def _errors(report):
    return {(i["code"], i["path"]) for i in report["issues"] if i["severity"] == "error"}


def test_e03_vanished_frozen_tensor_and_buffer_block():
    # A10-F1 case A: absence is never a pass, even when every common tensor is unchanged.
    model = _toy()
    before = params.snapshot(model, scope="synthetic")
    del model.vlm.lm_head
    del model.action_space.accel_mean
    after = params.snapshot(model, scope="synthetic")
    d = params.diff(before, after)
    assert d["only_in_before"] == ["action_space.accel_mean", "vlm.lm_head.weight"]
    assert d["changed"] == []
    # vlm.lm_head.weight has no counterpart (accel_mean is in (other)): not estimable.
    assert d["by_prefix"]["vlm."]["uncompared_tensors"] == 1
    assert d["by_prefix"]["vlm."]["changed_tensor_fraction"] is None
    assert d["by_prefix"]["vlm."]["changed_numel_fraction"] is None
    assert d["by_prefix"]["expert."]["changed_tensor_fraction"] == 0.0
    for manifest in (before, after):
        check = params.check_trainable_set(manifest, TRAINABLE, FROZEN, diff=d)
        assert check["status"] == "blocked"
        assert {("tensor_only_in_before", "action_space.accel_mean"),
                ("tensor_only_in_before", "vlm.lm_head.weight")} <= _errors(check)
    check = params.check_trainable_set(before, TRAINABLE, FROZEN, diff=d)
    assert check["explicit"]["vlm.lm_head.weight"]["changed"] is True
    check_after = params.check_trainable_set(after, TRAINABLE, FROZEN, diff=d)
    assert check_after["explicit"]["vlm.lm_head.weight"] == {
        "present": False, "classification": None, "requires_grad": None, "changed": True,
    }


def test_e03_wrapper_renamed_after_snapshot_blocks_and_fractions_are_not_zero():
    # A10-F1 case B: names that do not match are never compared, so nothing is "unchanged".
    inner = _toy()
    before = params.snapshot(inner, scope="synthetic")
    with torch.no_grad():
        inner.vlm.lm_head.weight.add_(1.0)
    wrapper = nn.Module()
    wrapper.expert_model = inner
    after = params.snapshot(wrapper, scope="synthetic")
    d = params.diff(before, after)
    assert d["changed"] == [] and len(d["only_in_before"]) == len(d["only_in_after"]) == 17
    for prefix in ("vlm.", "expert.", "action_in_proj.", "action_out_proj.", "(other)"):
        assert d["by_prefix"][prefix]["changed_tensor_fraction"] is None, prefix
        assert d["by_prefix"][prefix]["changed_numel_fraction"] is None, prefix
    assert d["by_prefix"]["vlm."]["uncompared_tensors"] == 10
    check = params.check_trainable_set(before, TRAINABLE, FROZEN, diff=d)
    assert check["status"] == "blocked"
    codes = {code for code, _ in _errors(check)}
    assert {"tensor_only_in_before", "tensor_only_in_after"} <= codes
    assert check["explicit"]["vlm.lm_head.weight"]["changed"] is True


def test_e03_requires_grad_flip_without_byte_change_blocks():
    # A10-F1 case C: the before snapshot says frozen; the step ran with the weight trainable.
    model = _toy()
    before = params.snapshot(model, scope="synthetic")
    model.vlm.lm_head.weight.requires_grad_(True)
    d = params.diff(before, params.snapshot(model, scope="synthetic"))
    assert d["requires_grad_changed"] == ["vlm.lm_head.weight"]
    check = params.check_trainable_set(before, TRAINABLE, FROZEN, diff=d)
    assert check["status"] == "blocked"
    assert ("requires_grad_changed", "vlm.lm_head.weight") in _errors(check)


def test_e03_kind_change_blocks():
    snap = params.snapshot(_toy(), scope="synthetic")
    other = json.loads(json.dumps(snap))
    other["tensors"]["action_space.accel_mean"]["kind"] = "parameter"
    d = params.diff(snap, other)
    assert d["kind_changed"] == ["action_space.accel_mean"]
    check = params.check_trainable_set(snap, TRAINABLE, FROZEN, diff=d)
    assert ("kind_changed", "action_space.accel_mean") in _errors(check)


def test_trainable_set_rejects_diff_without_drift_lists():
    snap = params.snapshot(_toy(), scope="synthetic")
    d = params.diff(snap, snap)
    for key in ("only_in_before", "only_in_after", "requires_grad_changed", "kind_changed"):
        broken = dict(d)
        broken[key] = None
        check = params.check_trainable_set(snap, TRAINABLE, FROZEN, diff=broken)
        assert ("diff_invalid", "diff") in _errors(check), key


def test_snapshot_script_diff_refuses_empty_prefix(tmp_path, capsys):
    # A10-F6: a bad --prefix is a refusal (exit 2 with a JSON error), not a traceback.
    snap = params.snapshot({"a.x": torch.tensor([1.0])}, scope="synthetic")
    (tmp_path / "a.json").write_text(json.dumps(snap))
    code = s4_snapshot_params.main(["diff", str(tmp_path / "a.json"), str(tmp_path / "a.json"),
                                    "--prefix", ""])
    assert code == 2
    err = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert err["issues"][0]["code"] == "argument_invalid"


def test_snapshot_script_output_in_missing_directory_is_refused(tmp_path, capsys):
    snap = params.snapshot({"a.x": torch.tensor([1.0])}, scope="synthetic")
    (tmp_path / "a.json").write_text(json.dumps(snap))
    code = s4_snapshot_params.main(["diff", str(tmp_path / "a.json"), str(tmp_path / "a.json"),
                                    "--output", str(tmp_path / "missing" / "d.json")])
    assert code == 2
    err = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert err["issues"][0]["code"] == "output_write_failed"


def _release_dir(tmp_path, tensors):
    release = tmp_path / "release"
    release.mkdir()
    save_file(tensors, str(release / "model.safetensors"))
    return release


def _links(bundle, release, *extra):
    return s4_snapshot_params.build([
        "links", "--bundle-root", str(bundle), "--release-root", str(release),
        "--scope", "synthetic", "--file", "model.safetensors", *extra,
    ])


def test_bundle_links_record_symlinked_shards_bound_to_release_bytes(tmp_path):
    # A10-F7: a converted Gym bundle symlinks the release shards (R3 section 2).
    tmp_path = tmp_path.resolve()
    release = _release_dir(tmp_path, {"expert.w": torch.tensor([1.0, 2.0])})
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.safetensors").symlink_to(release / "model.safetensors")
    record = _links(bundle, release)
    assert record["artifact_kind"] == "s4_bundle_links" and record["status"] == "passed"
    raw = (release / "model.safetensors").read_bytes()
    assert record["release_files"] == {"model.safetensors": _sha(raw)}
    entry = record["entries"]["model.safetensors"]
    assert entry["kind"] == "symlink"
    assert entry["normalized_target"] == str(release / "model.safetensors")
    # A relative link to the same file is the same target.
    rel_bundle = tmp_path / "bundle_rel"
    rel_bundle.mkdir()
    (rel_bundle / "model.safetensors").symlink_to("../release/model.safetensors")
    assert _links(rel_bundle, release)["status"] == "passed"
    # A byte copy is accepted by its hash.
    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / "model.safetensors").write_bytes(raw)
    copied = _links(copy, release)
    assert copied["status"] == "passed"
    assert copied["entries"]["model.safetensors"]["kind"] == "regular_file"


def test_bundle_links_negative_controls(tmp_path):
    tmp_path = tmp_path.resolve()
    release = _release_dir(tmp_path, {"expert.w": torch.tensor([1.0, 2.0])})
    other = tmp_path / "other"
    other.mkdir()
    save_file({"expert.w": torch.tensor([1.0, 2.0])}, str(other / "model.safetensors"))
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    (wrong / "model.safetensors").symlink_to(other / "model.safetensors")
    record = _links(wrong, release)
    assert record["status"] == "blocked"
    assert [i["code"] for i in record["issues"]] == ["bundle_link_target_mismatch"]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert [i["code"] for i in _links(empty, release)["issues"]] == ["bundle_file_missing"]
    changed = tmp_path / "changed"
    changed.mkdir()
    save_file({"expert.w": torch.tensor([1.0, 2.5])}, str(changed / "model.safetensors"))
    assert [i["code"] for i in _links(changed, release)["issues"]] == ["bundle_file_bytes_differ"]
    linked_root = tmp_path / "linked_root"
    linked_root.symlink_to(empty)
    with pytest.raises(s4_snapshot_params.CliError):
        _links(linked_root, release)


def test_bundle_links_refuse_lexical_match_through_symlinked_directory(tmp_path):
    # A10R (re-verify probe case C): '../evil/../release/model.safetensors' normalizes to the
    # release file, but ckpt/evil is a symlink, so the link physically reaches other bytes.
    tmp_path = tmp_path.resolve()
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    release = _release_dir(ckpt, {"expert.w": torch.tensor([1.0, 2.0])})
    (tmp_path / "elsewhere" / "x").mkdir(parents=True)
    (tmp_path / "elsewhere" / "release").mkdir()
    save_file({"expert.w": torch.tensor([9.0, 9.0])},
              str(tmp_path / "elsewhere" / "release" / "model.safetensors"))
    (ckpt / "evil").symlink_to(tmp_path / "elsewhere" / "x")
    bundle = ckpt / "bundle"
    bundle.mkdir()
    (bundle / "model.safetensors").symlink_to("../evil/../release/model.safetensors")
    record = _links(bundle, release)
    assert record["status"] == "blocked"
    assert "bundle_link_target_invalid" in [i["code"] for i in record["issues"]]
    assert record["entries"]["model.safetensors"]["resolved_target"] == str(
        tmp_path / "elsewhere" / "release" / "model.safetensors")
    release_manifest = s4_snapshot_params.build([
        "manifest", "--root", str(release), "--scope", "synthetic",
        "--file", "model.safetensors"])
    report = export_audit.check_zero_update_export(
        release=release_manifest, export=release_manifest, inference=None,
        optimizer_steps_applied=0, hf_config_is_vlm=True, gym_bundle=record)
    assert report["requirements"]["gym_bundle_matches_release"] == "blocked"
    # Even with the record's status forged to passed, the audit re-checks the target.
    forged = {**record, "status": "passed", "issues": []}
    report = export_audit.check_zero_update_export(
        release=release_manifest, export=release_manifest, inference=None,
        optimizer_steps_applied=0, hf_config_is_vlm=True, gym_bundle=forged)
    assert report["requirements"]["gym_bundle_matches_release"] == "blocked"


@pytest.mark.parametrize("make_target", [
    lambda release: f"{release}/./model.safetensors",
    lambda release: f"{release}//model.safetensors",
    lambda release: f"{release.parent}/release/../release/model.safetensors",
])
def test_bundle_links_refuse_non_canonical_targets_to_the_right_file(tmp_path, make_target):
    # A10R: '.', empty and inner '..' components are refused even when they reach the release.
    tmp_path = tmp_path.resolve()
    release = _release_dir(tmp_path, {"expert.w": torch.tensor([1.0, 2.0])})
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.safetensors").symlink_to(make_target(release))
    record = _links(bundle, release)
    assert record["status"] == "blocked"
    assert [i["code"] for i in record["issues"]] == ["bundle_link_target_invalid"]


def test_bundle_links_physical_resolution_must_be_the_release_file(tmp_path, monkeypatch):
    # A10R: a canonical target is still compared physically (realpath and st_dev/st_ino), so a
    # swap between the lexical check and the stat (or any resolver disagreement) is refused.
    tmp_path = tmp_path.resolve()
    release = _release_dir(tmp_path, {"expert.w": torch.tensor([1.0, 2.0])})
    other = tmp_path / "other"
    other.mkdir()
    save_file({"expert.w": torch.tensor([9.0, 9.0])}, str(other / "model.safetensors"))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.safetensors").symlink_to(release / "model.safetensors")
    passed = _links(bundle, release)
    assert passed["status"] == "passed"
    assert passed["entries"]["model.safetensors"]["resolved_target"] == str(
        release / "model.safetensors")
    monkeypatch.setattr(s4_snapshot_params, "_resolve_link",
                        lambda directory_fd, bundle_root, name, expected: (
                            str(other / "model.safetensors"), False))
    record = _links(bundle, release)
    assert record["status"] == "blocked"
    assert [i["code"] for i in record["issues"]] == ["bundle_link_physical_mismatch"]
