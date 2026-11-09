"""S4 launch glue: compiled registration overrides -> capture host-launch argv (argv only).

Tickets: S4.3 (qualified capture without forking upstream), E15 (the effective configuration is
part of the run identity), lane-r3 section 3 step 6 and section 6 H1/H4, build-spec B3 item
(2). Serves S4.1-S4.4 [C] (G9, G7) under D-024. Agent-authored evaluator engineering (D-018).

``build_launch`` turns ``teacher_rl.registration.compile_overrides`` output into the hashed
argv of ``teacher_rl.capture.host_launch`` (H4: the upstream host CLI with the capture
entrypoint)::

    uv run --no-sync --all-packages python -B -m teacher_rl.capture.host_launch
        --action-authority-ref <ref> --capture-tree <dir> --capture-manifest <file>
        --capture-config <file> --worker-credentials <mode> -- <Hydra overrides>

That hashed argv is not the exact run command: ``host_launch.main`` also requires the four
``host_launch.AUTHORITY_FLAGS`` pairs ``--launch-spec <this spec as JSON> --registration
<registration JSON> --decisions <decision log> --decision-id <D-NNN>``, inserted before
``--``. They point at the authority, so they are excluded from the hashed argv and the
launcher compares only the remaining flags and overrides with the spec. ``launch_hash`` is
``identity_hash(host_launch.launch_body(spec))``, the same body the launcher re-derives, so the
two cannot drift. ``authority_request`` (outside the hashed body; ``None`` when blocked) lists
the exact decision-entry lines Huy must write in the decision named by ``--decision-id``::

    authorizes: s4_launch_plan sha256:<launch_hash>
    authorizes: s4_registration sha256:<identity_hash(registration)>
    records: acquire:nurec_scenes
    records: terms:nurec_scenes

Writing those lines is Huy's recorded authority; this module never writes or proposes a
decision. The spec is run in the AlpaGym checkout with ``PYTHONPATH=<capture tree>`` (host_launch refuses any other
value) and ``HF_TOKEN`` supplied from a secret file at run time for the host CLI's own
preflight (never in argv). Routes:

- ``hydra_only``: the compiled Hydra overrides as they are.
- ``h1_resolved_toml`` (lane-r3 section 6 H1): Cosmos fields absent from the AlpaGym schema
  (``train.seed``, ``train.optm_warmup_start_factor``, ``train.train_policy.dataloader_seed``)
  are merged into the host-written ``<run_dir>/cosmos_config.toml`` and the host CLI is
  launched with ``execution.resolved_config_path='<run_dir>/resolved_config.yaml'`` so it loads
  the frozen config and skips ``write_run_artifacts`` (alpagym ``cli.py:28-41``); ``launch_all``
  then reads the harness-written TOML (``run_lifecycle.py:438-439``). The base TOML bytes and the
  sha256 of the host-composed ``resolved_config.yaml`` are required inputs (they exist only on
  the host); the base TOML must belong to ``run_dir`` (``train.output_dir == <run_dir>/cosmos``,
  ``run_artifacts.py``) and carry ``trainer_type = "alpagym_grpo"``. TOML has no null, so a
  registered value equal to the Cosmos default is left absent (``compile_overrides`` records it
  as ``upstream_default_matches``) and the effective-config hash still binds it.

  Because the frozen config makes the argv overrides inert on this route, the base TOML must
  already carry every registered value (review B3-01): each compiled ``cosmos.<key>`` Hydra
  override and ``policy.model.path`` is mapped to its TOML key with the pinned renames
  (``run_artifacts.py:100-131``: ``num_epochs`` -> ``epoch``,
  ``log_training_metrics_every_n_steps`` -> ``log_interval``, ``grpo_ratio_clip_low/high`` ->
  ``epsilon_low/high``, ``grpo_optimization_iterations`` -> ``mu_iterations``,
  ``policy.model.path`` -> ``policy.model_name_or_path``) and must be present with the same
  type and value (``base_toml_registered_key_missing`` / ``..._value_mismatch``); a
  registered Cosmos default left absent must be absent or equal. Every base TOML key and
  string is secret-scanned (rule 5, review B3-12) and a credential-like one blocks without
  being echoed. The run identity is ``effective_config_hash`` (registration) plus
  ``launch_hash``, which binds the exact base TOML bytes (``base_cosmos_toml_sha256``).

Both routes (review B3-04, B3-08):

- ``alpasim_cache_dir`` (required, portable absolute path) is compiled as
  ``alpasim.checkout_cache_dir='<dir>'`` so AlpaGym's content-addressed AlpaSim checkout lands
  where ``teacher_rl.ops`` verifies it (``alpasim_dependency.py:37-41,248-253``; agent
  technical default (D-018) choosing shipped behaviour over ``alpasim.repo_path``, which syncs
  an editable venv in place and skips ``compile-protos``). On H1 the frozen
  ``resolved_config.yaml`` must contain it (its hash is a host input; qualify_on_host).
- ``nurec_authorization_refs`` (required ``{"terms", "acquire"}``): the host CLI's
  ``validate_huggingface_access`` checks ``HF_TOKEN`` against the gated
  ``nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`` dataset (alpagym ``cli.py:54-57``,
  ``huggingface_validation.py``) and the Wizard downloads the scenes at launch, so the token
  never reaches the host CLI without Huy's recorded NuRec terms and acquisition references.

The TOML writer is minimal and fails closed: tables, arrays of tables, strings, booleans,
integers, finite floats and arrays of those; dates/times or mixed arrays are refused, and the
emitted bytes must parse back (``tomllib``) to exactly the merged mapping.

Every launch spec records ``effective_config_hash`` (registration, pins, route, overrides),
the TOML sha256 and ``launch_hash`` over argv, cwd, environment names/values (secret names
only), files to write and both config hashes.

Verification level per function:

- ``dump_toml``, ``merge_cosmos_toml``, ``build_launch``: **L1** (literal argv, TOML bytes and
  negative controls in ``tests/test_s4_launch.py``); the argv targets pinned upstream
  interfaces (**L2**). That Hydra composes the overrides, that ``launch_all`` round-trips the
  unknown TOML keys and that the composed ``resolved_config.yaml`` matches these overrides are
  **L3** (``qualify_on_host``).

Agent technical defaults (D-018): ``ENV_ALLOWLIST`` (``PATH``, ``HOME``, ``LANG``, ``LC_ALL``,
``HF_HOME`` copied from the operator's environment) and ``SECRET_ENV`` (``HF_TOKEN`` read from
the ``hf_token`` secret file at run time, for the host CLI's Hugging Face preflight only).

Never: executes anything (``executes: False``; tests only inspect argv), writes a file,
embeds or forwards a credential value, or chooses an owner value, worker-credential mode or
licence/acquisition reference.
"""

import hashlib
import json
import math
import re
import tomllib
from pathlib import Path, PurePosixPath

from gate.s2_identity import identity_hash
from gate.s2_runtime import is_placeholder, is_portable_absolute, secret_like
from teacher_rl.capture import host_launch
from teacher_rl.registration import compile_overrides

ARTIFACT_KIND = "s4_launch_spec"
SCHEMA_VERSION = 1
H1_ROUTE = "h1_resolved_toml"
HYDRA_ROUTE = "hydra_only"
RUN_DIR_NAME = re.compile(r"\d{8}T\d{6}Z-[0-9a-f]{32}")  # run_artifacts.py:22-39 (INFERRED)
ENV_ALLOWLIST = ["PATH", "HOME", "LANG", "LC_ALL", "HF_HOME"]
SECRET_ENV = {"HF_TOKEN": "hf_token"}
_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")
_HEX64 = re.compile(r"[0-9a-f]{64}")
# alpagym run_artifacts.py:100-107 and 128-131 (pinned 972d160): Hydra name -> TOML name.
COSMOS_TOML_RENAMES = {
    "train.num_epochs": "train.epoch",
    "logging.log_training_metrics_every_n_steps": "logging.log_interval",
    "train.train_policy.grpo_ratio_clip_low": "train.train_policy.epsilon_low",
    "train.train_policy.grpo_ratio_clip_high": "train.train_policy.epsilon_high",
    "train.train_policy.grpo_optimization_iterations": "train.train_policy.mu_iterations",
}
NUREC_AUTHORIZATION_KEYS = ("acquire", "terms")


class TomlError(ValueError):
    """A value the minimal TOML writer cannot represent exactly."""


def _issue(code, path, message, severity="blocking"):
    return {"code": code, "path": path, "severity": severity, "evidence_refs": [],
            "message": message}


def _producer():
    return {"module": "teacher_rl.launch",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


# --- TOML -------------------------------------------------------------------------


def _key(key):
    if type(key) is not str or not key:
        raise TomlError("keys are nonempty strings")
    return key if _BARE_KEY.fullmatch(key) else _string(key)


def _string(value):
    return json.dumps(value, ensure_ascii=True).replace("\x7f", "\\u007f")


def _is_table_array(value):
    return type(value) is list and bool(value) and all(type(item) is dict for item in value)


def _value(value, path):
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise TomlError(f"{path}: non-finite float")
        return repr(value)
    if type(value) is str:
        return _string(value)
    if type(value) is list:
        if any(type(item) is dict for item in value):
            raise TomlError(f"{path}: tables inside an inline array are not written")
        return "[" + ", ".join(_value(item, f"{path}[]") for item in value) + "]"
    raise TomlError(f"{path}: {type(value).__name__} is not representable (dates, null)")


def _emit(lines, path, table, header):
    if header is not None:
        lines.append(header)
    subtables = []
    for key in sorted(table):
        value = table[key]
        dotted = [*path, _key(key)]
        if type(value) is dict or _is_table_array(value):
            subtables.append((dotted, value))
        else:
            lines.append(f"{_key(key)} = {_value(value, '.'.join(dotted))}")
    for dotted, value in subtables:
        name = ".".join(dotted)
        if type(value) is dict:
            _emit(lines, dotted, value, f"[{name}]")
        else:
            for item in value:
                _emit(lines, dotted, item, f"[[{name}]]")


def dump_toml(mapping):
    """Canonical TOML bytes (sorted keys) that parse back to exactly ``mapping`` (L1)."""
    if type(mapping) is not dict:
        raise TomlError("the document is a table")
    lines = []
    _emit(lines, [], mapping, None)
    text = "\n".join(lines) + "\n"
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise TomlError(f"emitted TOML does not parse: {exc}") from None
    if parsed != mapping or json.dumps(parsed, sort_keys=True) != json.dumps(mapping,
                                                                             sort_keys=True):
        raise TomlError("emitted TOML does not round-trip to the same mapping")
    return text.encode()


def merge_cosmos_toml(base_bytes, overrides):
    """Merge dotted Cosmos overrides into parsed base TOML; return merged mapping and bytes."""
    if type(base_bytes) is not bytes:
        raise TomlError("base TOML is bytes")
    try:
        merged = tomllib.loads(base_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise TomlError(f"base TOML does not parse: {exc}") from None
    changes = []
    for dotted in sorted(overrides):
        value = overrides[dotted]
        if value is None:
            raise TomlError(f"{dotted}: TOML has no null; leave the key absent")
        parts = dotted.split(".")
        node = merged
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if type(child) is not dict:
                raise TomlError(f"{dotted}: {part} is not a table in the base TOML")
            node = child
        previous = node.get(parts[-1])
        if type(previous) is dict:
            raise TomlError(f"{dotted}: refuses to replace a table with a value")
        node[parts[-1]] = value
        changes.append({"key": dotted, "previous": previous, "value": value,
                        "action": "added" if previous is None else "replaced"})
    data = dump_toml(merged)
    return {"merged": merged, "bytes": data, "sha256": hashlib.sha256(data).hexdigest(),
            "changes": changes, "base_sha256": hashlib.sha256(base_bytes).hexdigest()}


# --- launch -----------------------------------------------------------------------


def _absolute(value):
    if type(value) is not str or not value.startswith("/") or "\x00" in value:
        return False
    pure = PurePosixPath(value)
    return str(pure) == value and ".." not in pure.parts and value != "/"


def toml_key(hydra_key):
    """The cosmos_config.toml key a compiled Hydra key lands on, or None (L2, pinned)."""
    if hydra_key == "policy.model.path":
        return "policy.model_name_or_path"
    if type(hydra_key) is str and hydra_key.startswith("cosmos."):
        key = hydra_key[len("cosmos."):]
        return COSMOS_TOML_RENAMES.get(key, key)
    return None


def _lookup(mapping, dotted):
    node = mapping
    for part in dotted.split("."):
        if type(node) is not dict or part not in node:
            return False, None
        node = node[part]
    return True, node


def _same(left, right):
    """Equal with equal JSON types (tomllib bool/int/float are distinct; no 1 == 1.0)."""
    if type(left) is not type(right):
        return False
    if type(left) is list:
        return len(left) == len(right) and all(map(_same, left, right))
    return left == right


def _registered_value_issues(compiled, base, issues):
    """Every registered Cosmos value must already be in the host-written TOML (B3-01)."""
    checked = []
    for entry in compiled["mapping"]:
        key = toml_key(entry["key"]) if entry["category"] == "hydra_key" else None
        default_left_absent = (entry["category"] == "cosmos_toml_key"
                               and entry.get("applied") == "upstream_default_matches")
        if key is None and not default_left_absent:
            continue
        key = key or entry["key"]
        found, value = _lookup(base, key)
        checked.append(key)
        if default_left_absent:
            if found and not _same(value, entry["value"]):
                issues.append(_issue("base_toml_registered_value_mismatch",
                                     f"base_cosmos_toml.{key}",
                                     f"registered {entry['field']} is the Cosmos default; the "
                                     "base TOML sets another value"))
        elif entry["value"] is None:
            if found:  # the host drops null fields (run_artifacts.py:138)
                issues.append(_issue("base_toml_registered_value_mismatch",
                                     f"base_cosmos_toml.{key}",
                                     f"registered {entry['field']} is null; the base TOML sets it"))
        elif not found:
            issues.append(_issue("base_toml_registered_key_missing", f"base_cosmos_toml.{key}",
                                 f"the host-written TOML lacks registered {entry['field']} "
                                 f"({entry['key']}); on H1 the argv override has no effect"))
        elif not _same(value, entry["value"]):
            issues.append(_issue("base_toml_registered_value_mismatch",
                                 f"base_cosmos_toml.{key}",
                                 f"registered {entry['field']} ({entry['key']}) differs from the "
                                 "host-written TOML that launch_all reads on H1"))
    return sorted(set(checked))


def _toml_strings(value):
    if type(value) is str:
        yield value
    elif type(value) is dict:
        for key, item in value.items():
            yield key
            yield from _toml_strings(item)
    elif type(value) is list:
        for item in value:
            yield from _toml_strings(item)


def _nurec_issues(refs, issues):
    good = type(refs) is dict and set(refs) == set(NUREC_AUTHORIZATION_KEYS) and all(
        type(v) is str and v.strip() and not is_placeholder(v) and not secret_like(v)
        for v in refs.values())
    if not good:
        issues.append(_issue("nurec_authorization_missing", "nurec_authorization_refs",
                             "HF_TOKEN reaches the host CLI's NuRec dataset check and the "
                             "Wizard's scene download: recorded, non-placeholder, non-secret "
                             f"{list(NUREC_AUTHORIZATION_KEYS)} references are required"))


def _h1_inputs(compiled, run_dir, base_toml, resolved_sha, issues):
    run_root = None
    for entry in compiled["mapping"]:
        if entry["key"] == "run_root":
            run_root = entry["value"]
    if not _absolute(run_dir) or PurePosixPath(run_dir).parent != PurePosixPath(run_root) or (
            RUN_DIR_NAME.fullmatch(PurePosixPath(run_dir).name) is None):
        issues.append(_issue("run_dir_invalid", "run_dir", "H1 needs the host-created run "
                             f"directory {run_root}/<YYYYMMDDTHHMMSSZ>-<uuid4 hex>"))
    if type(resolved_sha) is not str or _HEX64.fullmatch(resolved_sha) is None:
        issues.append(_issue("resolved_config_hash_missing", "resolved_config_sha256",
                             "sha256 of the host-composed resolved_config.yaml is required"))
    if type(base_toml) is not bytes:
        issues.append(_issue("base_toml_missing", "base_cosmos_toml",
                             "the host-written cosmos_config.toml bytes are required"))
        return None
    if issues:
        return None
    try:
        base = tomllib.loads(base_toml.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        issues.append(_issue("toml_unrepresentable", "base_cosmos_toml", str(exc)))
        return None
    if any(secret_like(text) for text in _toml_strings(base)):
        issues.append(_issue("secret_in_base_toml", "base_cosmos_toml",
                             "a credential-like key or value is in the base TOML (not echoed)"))
        return None
    try:
        toml = merge_cosmos_toml(base_toml, compiled["cosmos_toml_overrides"])
    except TomlError as exc:
        issues.append(_issue("toml_unrepresentable", "base_cosmos_toml", str(exc)))
        return None
    toml["registered_keys_checked"] = _registered_value_issues(compiled, base, issues)
    train = toml["merged"].get("train", {})
    policy = train.get("train_policy", {}) if type(train) is dict else {}
    if type(policy) is not dict or policy.get("trainer_type") != "alpagym_grpo":
        issues.append(_issue("base_toml_unexpected", "base_cosmos_toml",
                             "train.train_policy.trainer_type is not alpagym_grpo "
                             "(run_artifacts.py:117-124)"))
    if train.get("output_dir") != f"{run_dir}/cosmos":
        issues.append(_issue("base_toml_run_dir_mismatch", "base_cosmos_toml",
                             f"train.output_dir must be {run_dir}/cosmos (this run directory)"))
    return toml


def build_launch(registration, *, alpagym_checkout, capture_tree, capture_manifest,
                 capture_config, worker_credentials, action_authority_ref, run_dir=None,
                 base_cosmos_toml=None, resolved_config_sha256=None, alpasim_cache_dir=None,
                 nurec_authorization_refs=None):
    """Compile the host-launch argv for one registered run (argv only; never executed).

    ``alpasim_cache_dir`` and ``nurec_authorization_refs`` have no usable default: ``None``
    blocks (``alpasim_cache_dir_unset``, ``nurec_authorization_missing``).
    """
    issues = []
    compiled = compile_overrides(registration)
    if compiled["status"] != "passed":
        issues.append(_issue("registration_not_compiled", "registration",
                             f"{sorted({i['code'] for i in compiled['issues']})}"))
    for name, value in (("alpagym_checkout", alpagym_checkout), ("capture_tree", capture_tree),
                        ("capture_manifest", capture_manifest),
                        ("capture_config", capture_config)):
        if not _absolute(value):
            issues.append(_issue("path_not_absolute", name, "normalized absolute path"))
    if worker_credentials not in host_launch.WORKER_CREDENTIAL_MODES:
        issues.append(_issue("worker_credentials_unset", "worker_credentials",
                             f"one of {host_launch.WORKER_CREDENTIAL_MODES} (no default)"))
    try:
        host_launch._authority(action_authority_ref)
    except host_launch.LaunchRefused as exc:
        issues.append(_issue("action_authority_missing", "action_authority_ref", str(exc)))
    if not is_portable_absolute(alpasim_cache_dir):
        issues.append(_issue("alpasim_cache_dir_unset", "alpasim_cache_dir",
                             "the AlpaSim checkout cache directory teacher_rl.ops verifies "
                             "(portable absolute path; no default)"))
    _nurec_issues(nurec_authorization_refs, issues)
    route = compiled.get("effective_config", {}).get("config_route")
    toml = None
    overrides = list(compiled.get("hydra", {}).get("overrides", []))
    overrides.append(f"alpasim.checkout_cache_dir='{alpasim_cache_dir}'")
    if compiled["status"] == "passed" and route == H1_ROUTE:
        toml = _h1_inputs(compiled, run_dir, base_cosmos_toml, resolved_config_sha256, issues)
        overrides.append(f"execution.resolved_config_path='{run_dir}/resolved_config.yaml'")
    elif compiled["status"] == "passed" and any(
            v is not None for v in (run_dir, base_cosmos_toml, resolved_config_sha256)):
        issues.append(_issue("h1_inputs_on_hydra_route", "run_dir",
                             "run_dir/base TOML/resolved config hash belong to the H1 route"))
    if compiled["status"] == "passed":
        try:
            host_launch.check_run_root(overrides)
        except host_launch.LaunchRefused as exc:
            issues.append(_issue("run_root_invalid", "overrides", str(exc)))
    argv = ["uv", "run", "--no-sync", "--all-packages", "python", "-B", "-m",
            host_launch.__name__, "--action-authority-ref", str(action_authority_ref),
            "--capture-tree", str(capture_tree), "--capture-manifest", str(capture_manifest),
            "--capture-config", str(capture_config), "--worker-credentials",
            str(worker_credentials), "--", *overrides]
    if any(secret_like(item) for item in argv):
        issues.append(_issue("secret_in_argv", "argv", "credentials never appear in argv"))
    issues.sort(key=lambda i: (i["path"], i["code"]))
    blocked = bool(issues)
    files = [] if toml is None or blocked else [
        {"path": f"{run_dir}/cosmos_config.toml", "sha256": toml["sha256"],
         "bytes": len(toml["bytes"]), "content": toml["bytes"].decode()}]
    nurec = None if blocked else dict(sorted(nurec_authorization_refs.items()))
    spec = {
        "artifact_kind": ARTIFACT_KIND, "schema_version": SCHEMA_VERSION,
        "scope": registration.get("scope") if type(registration) is dict else None,
        "producer": _producer(),
        "input_hashes": {"registration": compiled["inputs"][0]["identity_hash"],
                         "base_cosmos_toml": None if type(base_cosmos_toml) is not bytes
                         else hashlib.sha256(base_cosmos_toml).hexdigest()},
        "alpasim_cache_dir": None if blocked else alpasim_cache_dir,
        "nurec_authorizations": nurec,
        "units": {"bytes": "bytes"}, "status": "blocked" if blocked else "passed",
        "issues": issues, "level": "L2", "execution_status": "qualify_on_host",
        "executes": False, "route": route,
        "argv": None if blocked else argv,
        "cwd": None if blocked else alpagym_checkout,
        "env_set": None if blocked else {"PYTHONPATH": capture_tree},
        "env_allowlist": list(ENV_ALLOWLIST), "secret_env": dict(SECRET_ENV),
        "files_to_write": files,
        "toml": None if toml is None or blocked else {
            "merged": toml["merged"], "sha256": toml["sha256"], "changes": toml["changes"],
            "base_sha256": toml["base_sha256"],
            "registered_keys_checked": toml["registered_keys_checked"]},
        "effective_config_hash": compiled.get("effective_config_hash"),
        "resolved_config_sha256": resolved_config_sha256,
        "launch_hash": None,
        "authority_request": None,
        "qualify_on_host": [
            "resolved_config.yaml was composed from exactly these Hydra overrides (hash it)",
            "the controller's effective Cosmos config contains every TOML override",
            "PYTHONPATH reaches every Cosmos role and the capture entrypoint runs in each",
            ("H1: resolved_config.yaml carries alpasim.checkout_cache_dir and the registered "
             "values; the cached AlpaSim checkout is the one teacher_rl.ops verifies"),
            "the NuRec scenes the Wizard downloads are covered by the recorded references",
        ],
        "cloud_launch_authorized": False,
    }
    if not blocked:
        # One hashed body for both sides (G3R): the launcher re-derives exactly this.
        spec["launch_hash"] = identity_hash(host_launch.launch_body(spec))
        registration_sha = spec["input_hashes"]["registration"]
        spec["authority_request"] = [
            f"authorizes: {host_launch.LAUNCH_ARTIFACT_KIND} sha256:{spec['launch_hash']}",
            f"authorizes: {host_launch.REGISTRATION_ARTIFACT_KIND} sha256:{registration_sha}",
            *(f"records: {ref}" for ref in host_launch.NUREC_RECORDS)]
    return spec
