"""Zero-fork in-process capture adapter for AlpaGym's shipped trainer (lane-r3 §6 H4).

``entrypoint`` wraps upstream lookup bindings inside each Cosmos process and streams evidence to
``sink``; ``host_launch`` substitutes the entrypoint module in the host's launcher command.
Importing this package imports nothing else: every submodule is stdlib-only at import time and
is loaded explicitly by name, so it can run inside the AlpaGym environment via PYTHONPATH.
"""
