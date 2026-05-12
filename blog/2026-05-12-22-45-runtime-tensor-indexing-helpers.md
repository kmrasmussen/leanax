# Runtime Tensor Indexing Helpers

`TICKET-0067` turns the generated runtime path toward structured tensor codegen.

`LeanAX/RuntimeLLVM.lean` now has a `RuntimeTensor` helper layer for
deterministic scalar refs, row-major offsets, fixed-shape index enumeration,
tensor constants, dense products, reductions, transpose refs, ReLU refs, and
weighted checksums.

The first consumer is `generated-dense-runtime`: it now emits through the helper
path while still matching the existing golden LLVM MLIR and returning the same
`mlir-runner` checksum. This keeps the runtime surface stable before the next
tickets scale the same machinery to exact-shape MNIST forward and train-step
checksums.
