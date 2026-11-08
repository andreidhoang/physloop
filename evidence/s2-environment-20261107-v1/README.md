# S2 CPU environment reconstruction attempt

Agent-authored engineering evidence; no simulator, model, GPU or cloud run.

A genuinely new virtual environment was created using the installed CPython standard library. It contains only its seeded pip installer. Installing exact compatible wheel URLs and SHA-256 hashes selected from uv.lock failed on the first package because this host could not resolve files.pythonhosted.org. No installed project package was copied into the fresh environment. Tests were therefore not run there.

The older uv 0.8.17 fails earlier during macOS SystemConfiguration initialization, including with --offline. The official astral-sh/uv macOS arm64 CI artifact was downloaded by the connected service, but the authorized local file-transfer helper failed. No newer uv executable was materialized or executed.

The working project environment separately received protobuf 7.36.2 from a previously resolved cached distribution after all 62 RECORD hash/size entries passed. Its exact package block was reused from an existing uv lock and the requirement pinned. Other existing lock entries were unchanged. This fallback and the working environment's CPU test results do not establish a clean rebuild.

See protobuf-provenance.json for exact hashes/source, selected-lock-artifacts.json for the full selected dependency graph, requirements-from-lock.txt for the hash-constrained installation input, and the command logs plus rebuild-attempt.json for the failed rebuild. No registry URL, hash or transitive dependency was invented.

Result: clean_frozen_sync_verified=false; fresh_locked_install_verified=false; fresh_tests_executed=false. Both pyproject.toml and uv.lock were unchanged during the rebuild attempts.

Remaining action: a functioning uv with reachable locked artifact URLs must perform a fresh frozen sync and run the repository tests/rehearsal. The source lock must remain unchanged; cache-directory copying or repacking must not be labelled a clean rebuild.
