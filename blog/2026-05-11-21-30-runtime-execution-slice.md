# Runtime Execution Slice

`TICKET-0018` is now complete with a narrow but real external runtime path.

The current shell still does not provide IREE, XLA, or direct StableHLO runtime
tooling. Rather than mark runtime execution blocked forever, LeanAX now emits an
LLVM-dialect MLIR module for `affine-runtime` and runs it with the available
`mlir-runner` JIT.

The fixture mirrors the existing `affine` oracle inputs and computes the scalar
checksum `sum((x + bias) * (x + bias))`. The runner compares stdout from
`mlir-runner --entry-point-result=f32` against `94.25`, and the manifest has a
dedicated `runtime` outcome so this path is part of the normal Nix e2e gate.

This is not direct StableHLO execution yet. It is the first checked runtime
slice: LeanAX emits executable IR, an external runtime executes it, and the
result is compared against a deterministic fixture.
