# Pinned launch-source fixtures

These are unmodified public-source bytes from the two repositories and commits
listed in `manifest.json`, retrieved on 2026-11-07. They support source-interface
tests of the S2 command preparer; they are not an installed simulator or a runtime
qualification. No model weights, scenes, dataset metadata or renderer image is
included.

Each receipt records the upstream path, immutable source URL, Git blob SHA-1,
file size and SHA-256. All 23 files matched their returned Git blob identities
when saved. Python source filenames have `.txt` appended so the copies remain
test data rather than imported or reformatted project modules. File contents are
unchanged. The original Apache licenses are included under each repository's
directory.

The snapshots cover the load-bearing relationships: dev defaults and camera
selection; explicit scene selection and pre-deployment fetching; generated driver
configuration and driver entrypoint; renderer command composition; container
shell handling and conditional image builds; and the model's opt-in graph field.
See `docs/research/2026-11-07-pilot-launch-audit.md` for assembly, limits and the
separate unresolved asset/runtime inputs. The full repositories and dependency
trees are intentionally not vendored here.
