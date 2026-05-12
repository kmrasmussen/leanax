# Dense Runtime Fixture

`TICKET-0042` adds the second external runtime fixture.

`dense-runtime` emits LLVM-dialect MLIR for a fixed dense-layer checksum:
`1x4 @ 4x3 + 3`, reduced to a scalar sum of squares. The normal e2e runner
executes it with `mlir-runner` and checks the result `103.375`.

This is still not direct StableHLO runtime execution and not full MNIST runtime,
but it moves the runtime path from an affine checksum toward classifier-shaped
dense math.
