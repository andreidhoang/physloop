# Quaternion arithmetic reference

AlpaSim's pinned `utils_rs/Cargo.lock` resolves `glam` 0.29.3. Its annotated tag
resolves to commit `d36e7eeff05338c56c4aa8d59fc2615e7963b1b7`; full source-file
Git blob and SHA-256 receipts are in `provenance.json`. The Cargo files retain
AlpaSim's Apache-2.0 attribution and parent license. Glam offers MIT OR Apache-2.0;
both exact license files are retained here.

For every retained backend, `Quat::length_squared` delegates to
`Vec4::length_squared`, then to the backend's dot product. Quaternion component
storage is **xyzw**, while the ASL protobuf field order is **wxyz**.

The squared components and every addition use binary32 arithmetic:

| Backend | Reduction of squared components |
|---|---|
| SSE2 | `(x² + z²) + (y² + w²)` |
| Scalar | `((x² + y²) + z²) + w²` |
| NEON | `(x² + y²) + (z² + w²)` |

The NEON helper uses `vmulq_f32` and `vaddvq_f32`. The official Arm ACLE
[intrinsic table](https://arm-software.github.io/acle/neon_intrinsics/advsimd.html),
page version `aef52c2b0944bf652443c1acd867dbb893af8dd0`, maps that reduction to
two pairwise floating additions. This supports the adjacent-pair expression.

The upstream `ensure_unit_quat` in the parent fixture rejects squared norms below
binary32 `1e-12` and accepts only binary32 `abs(norm² - 1) <= 1e-4`; it normalizes
accepted values. PhysLoop preserves original values and does **not** perform that
normalization. Checking all three reductions conservatively detects uncertainty
at the acceptance boundary without changing upstream's constants.

These are source-derived arithmetic references. They do not establish a particular
runtime compiler, selected backend, floating-point mode, service binary or measured
round-trip. No Rust extension was built or executed for this audit.
