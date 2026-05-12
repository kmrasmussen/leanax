# Runtime Shape Ops

`TICKET-0060` adds helper-generated runtime evidence for the shape-only
operation surface.

The new `broadcast-shape-runtime`, `reshape-shape-runtime`, and
`transpose-shape-runtime` cases use the runtime skeleton from `TICKET-0059`.
Each case scalarizes a tiny fixed-shape target tensor order and computes a
weighted checksum, so broadcast expansion, reshape storage order, and transpose
permutation drift show up as different scalar results.

The default e2e manifest runs all three through `mlir-runner` and checks
`46.0`, `91.0`, and `86.0`. The operation inventory now marks broadcast,
reshape, and transpose as covered, leaving reduce as the remaining unsupported
runtime operation in the derived-mask train-step surface.
