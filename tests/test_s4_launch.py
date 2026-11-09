"""Known answers and negative controls for teacher_rl.launch (B3 item 2; lane-r3 H1/H4).

Argv only: nothing is executed. The expected argv prefix is transcribed from
``teacher_rl/capture/host_launch.py`` (its argparse flags) and lane-r3 section 3 step 6; the
expected TOML is a hand-written literal whose sha256 is computed from the literal bytes.
"""

import datetime
import hashlib
import tomllib

import pytest

from scripts.rehearse_s4 import RUN_DIR_NAME, synthetic_base_cosmos_toml, synthetic_registration
from teacher_rl import launch
from teacher_rl.registration import compile_overrides

RUN_DIR = f"/data/alpagym-runs/{RUN_DIR_NAME}"
PREFIX = ["uv", "run", "--no-sync", "--all-packages", "python", "-B", "-m",
          "teacher_rl.capture.host_launch", "--action-authority-ref", "D-999 synthetic approval",
          "--capture-tree", "/data/capture-tree", "--capture-manifest", "/data/tree.json",
          "--capture-config", "/data/capture.json", "--worker-credentials", "strip_offline", "--"]
CACHE = "/data/cache/alpagym-alpasim"
CACHE_OVERRIDE = f"alpasim.checkout_cache_dir='{CACHE}'"
NUREC = {"terms": "SYNTHETIC-terms-nurec", "acquire": "SYNTHETIC-acquire-nurec"}
# Hand-written canonical dump (sorted keys, scalars before subtables) of the synthetic base
# TOML plus the one H1 override train.seed = 7.
MERGED_TOML = (
    'mode = "colocated"\n'
    '[policy]\n'
    'model_name_or_path = "/data/ckpt/alpagym-bundle"\n'
    '[rollout]\n'
    'n_generation = 2\n'
    'prefetch_rollout = false\n'
    '[train]\n'
    'compile = false\n'
    'epoch = 1\n'
    'max_num_steps = 3\n'
    'non_text = true\n'
    'optm_decay_ratio = 0.0\n'
    'optm_decay_type = "none"\n'
    'optm_lr = 0.0001\n'
    'optm_min_lr_factor = 0.0\n'
    'optm_warmup_steps = 1\n'
    f'output_dir = "{RUN_DIR}/cosmos"\n'
    'resume = false\n'
    'seed = 7\n'
    'train_batch_per_replica = 2\n'
    '[train.ckpt]\n'
    'enable_checkpoint = true\n'
    'export_safetensors = true\n'
    'max_keep = -1\n'
    'save_freq = 1\n'
    '[train.train_policy]\n'
    'allowed_outdated_steps = 0\n'
    'kl_beta = 0.0\n'
    'mini_batch = 1\n'
    'mu_iterations = 1\n'
    'on_policy = true\n'
    'reference_reset_interval = 0\n'
    'trainer_type = "alpagym_grpo"\n'
    'type = "grpo"\n'
    'use_remote_reward = false\n'
).encode()


def kwargs(**overrides):
    value = {"alpagym_checkout": "/data/src/alpagym", "capture_tree": "/data/capture-tree",
             "capture_manifest": "/data/tree.json", "capture_config": "/data/capture.json",
             "worker_credentials": "strip_offline",
             "action_authority_ref": "D-999 synthetic approval",
             "alpasim_cache_dir": CACHE, "nurec_authorization_refs": dict(NUREC)}
    value.update(overrides)
    return value


def h1(**overrides):
    value = kwargs(run_dir=RUN_DIR, base_cosmos_toml=synthetic_base_cosmos_toml(),
                   resolved_config_sha256="5" * 64)
    value.update(overrides)
    return value


def test_hydra_route_argv_is_the_capture_host_launch_with_compiled_overrides():
    registration = synthetic_registration(route="hydra_only")
    spec = launch.build_launch(registration, **kwargs())
    assert spec["status"] == "passed", spec["issues"]
    compiled = compile_overrides(registration)
    assert spec["argv"] == PREFIX + compiled["hydra"]["overrides"] + [CACHE_OVERRIDE]
    assert "run_root='/data/alpagym-runs'" in spec["argv"]
    assert spec["cwd"] == "/data/src/alpagym"
    assert spec["env_set"] == {"PYTHONPATH": "/data/capture-tree"}
    assert spec["secret_env"] == {"HF_TOKEN": "hf_token"}
    assert spec["files_to_write"] == [] and spec["toml"] is None
    assert spec["effective_config_hash"] == compiled["effective_config_hash"]
    assert spec["executes"] is False and spec["execution_status"] == "qualify_on_host"
    assert spec["launch_hash"] == launch.build_launch(registration, **kwargs())["launch_hash"]


def test_h1_route_writes_only_non_default_cosmos_keys_and_points_at_the_frozen_config():
    registration = synthetic_registration()
    spec = launch.build_launch(registration, **h1())
    assert spec["status"] == "passed", spec["issues"]
    compiled = compile_overrides(registration)
    # train_seed 7 differs from the Cosmos default (None); dataloader_seed 0 and start
    # factor 0.0 equal the defaults and stay absent (TOML has no null).
    assert compiled["cosmos_toml_overrides"] == {"train.seed": 7}
    assert spec["argv"] == PREFIX + compiled["hydra"]["overrides"] + [
        CACHE_OVERRIDE, f"execution.resolved_config_path='{RUN_DIR}/resolved_config.yaml'"]
    assert spec["toml"]["changes"] == [{"key": "train.seed", "previous": None, "value": 7,
                                        "action": "added"}]
    assert spec["files_to_write"] == [{
        "path": f"{RUN_DIR}/cosmos_config.toml",
        "sha256": hashlib.sha256(MERGED_TOML).hexdigest(), "bytes": len(MERGED_TOML),
        "content": MERGED_TOML.decode()}]
    assert spec["resolved_config_sha256"] == "5" * 64
    other = launch.build_launch(registration, **h1(resolved_config_sha256="6" * 64))
    assert other["launch_hash"] != spec["launch_hash"]


def test_dump_toml_round_trips_tables_arrays_and_quoted_keys():
    data = {"a": {"b": [1, 2], "c": "x\x7fy\n", "dotted.key": True, "f": 1e-06},
            "runs": [{"id": 1, "sub": {"z": "q"}}, {"id": 2}], "empty": {}, "s": "é"}
    text = launch.dump_toml(data)
    assert tomllib.loads(text.decode()) == data
    assert b'"dotted.key" = true' in text and b"[[runs]]" in text and b"[empty]" in text


@pytest.mark.parametrize("value", [{"x": float("nan")}, {"x": None}, {"x": [1, {"y": 2}]},
                                   {"x": datetime.date(2026, 11, 9)}, {"": 1}])
def test_dump_toml_refuses_unrepresentable_values(value):
    with pytest.raises(launch.TomlError):
        launch.dump_toml(value)


def test_merge_refuses_null_and_table_collisions():
    base = b"[train]\nseed = 1\n[train.sub]\nk = 1\n"
    with pytest.raises(launch.TomlError):
        launch.merge_cosmos_toml(base, {"train.seed": None})
    with pytest.raises(launch.TomlError):
        launch.merge_cosmos_toml(base, {"train.sub": 3})
    with pytest.raises(launch.TomlError):
        launch.merge_cosmos_toml(b"train = 1\n", {"train.seed": 3})
    with pytest.raises(launch.TomlError):
        launch.merge_cosmos_toml(b"when = 2026-11-09\n", {"train.seed": 3})
    merged = launch.merge_cosmos_toml(base, {"train.seed": 2})
    assert merged["changes"] == [{"key": "train.seed", "previous": 1, "value": 2,
                                  "action": "replaced"}]


@pytest.mark.parametrize("route, change, code", [
    ("hydra_only", {"action_authority_ref": "TBD"}, "action_authority_missing"),
    ("hydra_only", {"action_authority_ref": None}, "action_authority_missing"),
    ("hydra_only", {"worker_credentials": None}, "worker_credentials_unset"),
    ("hydra_only", {"capture_tree": "capture-tree"}, "path_not_absolute"),
    ("hydra_only", {"capture_config": "/data/../etc/x"}, "path_not_absolute"),
    ("hydra_only", {"run_dir": RUN_DIR}, "h1_inputs_on_hydra_route"),
    ("hydra_only", {"action_authority_ref": "hf_" + "a" * 34}, "secret_in_argv"),
    ("h1", {"run_dir": None}, "run_dir_invalid"),
    ("h1", {"run_dir": "/elsewhere/" + RUN_DIR_NAME}, "run_dir_invalid"),
    ("h1", {"run_dir": "/data/alpagym-runs/latest"}, "run_dir_invalid"),
    ("h1", {"resolved_config_sha256": None}, "resolved_config_hash_missing"),
    ("h1", {"base_cosmos_toml": None}, "base_toml_missing"),
    ("h1", {"base_cosmos_toml": synthetic_base_cosmos_toml().replace(b"alpagym_grpo", b"x")},
     "base_toml_unexpected"),
    ("h1", {"base_cosmos_toml": synthetic_base_cosmos_toml().replace(b"0" * 32, b"1" * 32)},
     "base_toml_run_dir_mismatch"),
    ("h1", {"base_cosmos_toml": b"[train\n"}, "toml_unrepresentable"),
    ("hydra_only", {"alpasim_cache_dir": None}, "alpasim_cache_dir_unset"),
    ("hydra_only", {"alpasim_cache_dir": "/data/it's"}, "alpasim_cache_dir_unset"),
    ("hydra_only", {"nurec_authorization_refs": None}, "nurec_authorization_missing"),
    ("hydra_only", {"nurec_authorization_refs": {"terms": "TBD", "acquire": "SYNTHETIC-a"}},
     "nurec_authorization_missing"),
    ("h1", {"nurec_authorization_refs": {"terms": "SYNTHETIC-t"}},
     "nurec_authorization_missing"),
    ("h1", {"nurec_authorization_refs": {"terms": "SYNTHETIC-t", "acquire": "hf_" + "a" * 34}},
     "nurec_authorization_missing"),
])
def test_launch_negative_controls(route, change, code):
    registration = synthetic_registration(route="hydra_only" if route == "hydra_only"
                                          else "h1_resolved_toml")
    args = kwargs(**change) if route == "hydra_only" else h1(**change)
    spec = launch.build_launch(registration, **args)
    assert spec["status"] == "blocked"
    assert code in {i["code"] for i in spec["issues"]}, spec["issues"]
    assert spec["argv"] is None and spec["launch_hash"] is None and spec["files_to_write"] == []


def test_blocked_registration_gives_no_argv():
    registration = synthetic_registration()
    registration["owner"].pop("reward_arm")
    spec = launch.build_launch(registration, **h1())
    assert spec["status"] == "blocked"
    assert "registration_not_compiled" in {i["code"] for i in spec["issues"]}
    assert spec["argv"] is None


# --- review B3-01: the host-written base TOML must carry every registered value ------------


def _contradict(old, new):
    base = synthetic_base_cosmos_toml()
    assert old in base
    return base.replace(old, new)


@pytest.mark.parametrize("old, new, key", [
    (b"optm_lr = 0.0001", b"optm_lr = 0.5", "train.optm_lr"),
    (b"mini_batch = 1", b"mini_batch = 64", "train.train_policy.mini_batch"),
    (b'model_name_or_path = "/data/ckpt/alpagym-bundle"',
     b'model_name_or_path = "/data/ckpt/some-other-model"', "policy.model_name_or_path"),
    (b"epoch = 1", b"epoch = 2", "train.epoch"),  # Hydra num_epochs -> TOML epoch
    (b"mu_iterations = 1", b"mu_iterations = 3", "train.train_policy.mu_iterations"),
    (b"n_generation = 2", b"n_generation = 6", "rollout.n_generation"),
    (b"export_safetensors = true", b"export_safetensors = false",
     "train.ckpt.export_safetensors"),
    (b"optm_decay_ratio = 0.0", b"optm_decay_ratio = 0", "train.optm_decay_ratio"),  # int
])
def test_h1_base_toml_contradicting_a_registered_value_blocks(old, new, key):
    spec = launch.build_launch(synthetic_registration(),
                               **h1(base_cosmos_toml=_contradict(old, new)))
    assert spec["status"] == "blocked"
    issue = next(i for i in spec["issues"] if i["code"] == "base_toml_registered_value_mismatch")
    assert issue["path"] == f"base_cosmos_toml.{key}"
    assert spec["argv"] is None and spec["launch_hash"] is None and spec["files_to_write"] == []


def test_h1_base_toml_missing_a_registered_key_blocks():
    spec = launch.build_launch(synthetic_registration(), **h1(
        base_cosmos_toml=_contradict(b"kl_beta = 0.0\n", b"")))
    assert spec["status"] == "blocked"
    assert {(i["code"], i["path"]) for i in spec["issues"]} == {
        ("base_toml_registered_key_missing", "base_cosmos_toml.train.train_policy.kl_beta")}


def test_h1_base_toml_contradicting_an_upstream_default_registration_blocks():
    # warmup.start_factor 0.0 is registered as the Cosmos default (left absent); a base TOML
    # carrying another value would be what launch_all reads.
    spec = launch.build_launch(synthetic_registration(), **h1(
        base_cosmos_toml=_contradict(b"optm_lr = 0.0001\n",
                                     b"optm_lr = 0.0001\noptm_warmup_start_factor = 0.5\n")))
    assert spec["status"] == "blocked"
    assert "base_cosmos_toml.train.optm_warmup_start_factor" in {
        i["path"] for i in spec["issues"] if i["code"] == "base_toml_registered_value_mismatch"}


def test_toml_key_mapping_follows_the_pinned_run_artifacts_renames():
    # run_artifacts.py:100-131 (pinned 972d160): the five renamed fields and the model path
    assert launch.toml_key("cosmos.train.num_epochs") == "train.epoch"
    assert launch.toml_key("cosmos.train.train_policy.grpo_optimization_iterations") == (
        "train.train_policy.mu_iterations")
    assert launch.toml_key("cosmos.train.train_policy.grpo_ratio_clip_low") == (
        "train.train_policy.epsilon_low")
    assert launch.toml_key("cosmos.train.train_policy.grpo_ratio_clip_high") == (
        "train.train_policy.epsilon_high")
    assert launch.toml_key("cosmos.logging.log_training_metrics_every_n_steps") == (
        "logging.log_interval")
    assert launch.toml_key("policy.model.path") == "policy.model_name_or_path"
    assert launch.toml_key("cosmos.train.optm_lr") == "train.optm_lr"
    assert launch.toml_key("dataset.scene_ids") is None


def test_launch_hash_binds_the_base_toml_bytes():
    spec = launch.build_launch(synthetic_registration(), **h1())
    reformatted = synthetic_base_cosmos_toml() + b"# comment only\n"
    other = launch.build_launch(synthetic_registration(), **h1(base_cosmos_toml=reformatted))
    assert other["status"] == "passed"
    assert other["files_to_write"] == spec["files_to_write"]  # same merged mapping
    assert other["launch_hash"] != spec["launch_hash"]


# --- review B3-12: credentials in the base TOML are refused, never copied -------------------


def test_secret_like_value_in_the_base_toml_blocks_and_is_not_echoed():
    canary = "hf_" + "Q" * 34  # synthetic canary, not a credential
    base = synthetic_base_cosmos_toml() + f'[logging]\napi_key = "{canary}"\n'.encode()
    spec = launch.build_launch(synthetic_registration(), **h1(base_cosmos_toml=base))
    assert spec["status"] == "blocked"
    assert "secret_in_base_toml" in {i["code"] for i in spec["issues"]}
    assert canary not in str(spec)


# --- review B3-04 / B3-08: AlpaSim cache directory and NuRec authorization ------------------


def test_cache_dir_and_nurec_authorizations_enter_the_launch_identity():
    spec = launch.build_launch(synthetic_registration(route="hydra_only"), **kwargs())
    assert spec["alpasim_cache_dir"] == CACHE
    assert spec["nurec_authorizations"] == NUREC
    other = launch.build_launch(synthetic_registration(route="hydra_only"), **kwargs(
        nurec_authorization_refs={"terms": "SYNTHETIC-terms-nurec", "acquire": "SYNTHETIC-b"}))
    assert other["launch_hash"] != spec["launch_hash"]
    # XDG_CACHE_HOME never reaches the host CLI; the cache directory is the explicit override
    assert "XDG_CACHE_HOME" not in launch.ENV_ALLOWLIST


# --- G3R: the authority request and one hashed body shared with host_launch ---------------


@pytest.mark.parametrize("route", ["hydra_only", "h1_resolved_toml"])
def test_authority_request_lists_the_exact_decision_entry_lines(route):
    from gate.s2_identity import identity_hash
    from teacher_rl.capture import host_launch

    registration = synthetic_registration(route=route)
    spec = launch.build_launch(registration, **(kwargs() if route == "hydra_only" else h1()))
    assert spec["status"] == "passed", spec["issues"]
    assert spec["authority_request"] == [
        f"authorizes: s4_launch_plan sha256:{spec['launch_hash']}",
        f"authorizes: s4_registration sha256:{identity_hash(registration)}",
        "records: acquire:nurec_scenes",
        "records: terms:nurec_scenes",
    ]
    # outside the hashed body: the hash re-derives through host_launch.launch_body
    body = host_launch.launch_body(spec)
    assert "authority_request" not in body
    assert spec["launch_hash"] == identity_hash(body)
    # the four authority flags are not in the hashed argv (host_launch inserts them)
    assert not set(host_launch.AUTHORITY_FLAGS) & set(spec["argv"])


def test_blocked_spec_has_no_authority_request():
    spec = launch.build_launch(synthetic_registration(route="hydra_only"),
                               **kwargs(alpasim_cache_dir=None))
    assert spec["status"] == "blocked" and spec["authority_request"] is None


def test_launch_hash_is_derived_through_host_launch_launch_body(monkeypatch):
    """No drift: build_launch hashes exactly what the launcher re-derives."""
    from gate.s2_identity import identity_hash
    from teacher_rl.capture import host_launch

    seen = []

    def marker(spec):
        seen.append(spec["argv"])
        return {"marker": len(seen)}

    monkeypatch.setattr(host_launch, "launch_body", marker)
    spec = launch.build_launch(synthetic_registration(route="hydra_only"), **kwargs())
    assert spec["status"] == "passed" and len(seen) == 1
    assert spec["launch_hash"] == identity_hash({"marker": 1})
