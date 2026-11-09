"""Hash-checked loader for the vendored pinned upstream sources (test reference only).

Every ``*.py.txt`` here is a byte-identical copy of an Apache-2.0 file at the commit in
``provenance.json`` (licences under ``licenses/``). ``load`` verifies the bytes, parses
the module, and compiles *only* the named top-level/class-member function or class
nodes into a caller-supplied restricted namespace. Upstream imports are never executed.
"""

import __future__

import ast
import hashlib
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).parent
PROVENANCE = json.loads((ROOT / "provenance.json").read_text())

SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "getattr",
        "hasattr",
        "int",
        "isinstance",
        "len",
        "list",
        "max",
        "min",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "zip",
        "__build_class__",
        "AssertionError",
        "FloatingPointError",
        "KeyError",
        "RuntimeError",
        "TypeError",
        "ValueError",
        "FileNotFoundError",
        "property",
        "staticmethod",
        "classmethod",
        # numpy's C methods (e.g. ndarray.any) resolve helpers through the calling
        # frame's __import__; the executed upstream nodes contain no import statements.
        "__import__",
    )
}


def source_bytes(name):
    entry = PROVENANCE["files"][name]
    content = (ROOT / name).read_bytes()
    assert hashlib.sha256(content).hexdigest() == entry["sha256"], f"{name} bytes changed"
    blob = b"blob %d\0" % len(content) + content
    assert hashlib.sha1(blob).hexdigest() == entry["git_blob_sha"]
    return content


def _find(tree, qualname):
    """Resolve ``A``, ``A.b`` (class member) or ``f.g`` (function nested in ``f``)."""
    kinds = (ast.FunctionDef, ast.ClassDef)
    node = None
    for part in qualname.split("."):
        if node is None:
            candidates = tree.body
        elif isinstance(node, ast.ClassDef):
            candidates = node.body
        else:
            candidates = [n for n in ast.walk(node) if n is not node]
        matches = [n for n in candidates if isinstance(n, kinds) and n.name == part]
        assert len(matches) == 1, f"{qualname}: expected one {part!r}, found {len(matches)}"
        node = matches[0]
    return node


def load(name, qualnames, namespace, *, strip_decorators=True):
    """Compile the named nodes of a vendored file into ``namespace`` and return it.

    A ``Class.method`` qualname yields a plain function bound under ``method``.
    Function decorators are removed (they name upstream runtime objects); class
    decorators are kept so dataclasses/enums behave as upstream defines them.
    """
    path = ROOT / name
    tree = ast.parse(source_bytes(name), filename=str(path))
    nodes = []
    for qualname in qualnames:
        node = _find(tree, qualname)
        if isinstance(node, ast.FunctionDef) and strip_decorators:
            node.decorator_list = []
        nodes.append(node)
    for node in nodes:
        assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(node))
    module_name = "s4_vendored_" + name.replace("/", "_").replace(".", "_")
    module = sys.modules.get(module_name) or types.ModuleType(module_name)
    sys.modules[module_name] = module  # dataclasses resolve cls.__module__ here
    module.__dict__.update(namespace)
    module.__dict__["__builtins__"] = namespace.get("__builtins__", SAFE_BUILTINS)
    code = compile(
        ast.Module(body=nodes, type_ignores=[]),
        str(path),
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    exec(code, module.__dict__)  # noqa: S102 - pinned, hash-checked, selected nodes only
    return module.__dict__
