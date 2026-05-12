# Direct Runtime Scale-Up Roadmap

This wave starts after `TICKET-0065`.

The generated runtime wave proved the LLVM route, the full operation surface,
and scaled generated representatives for forward and train-step semantics. The
remaining direct-runtime gap is scale, not concept:

- `runtime_readiness_v6` is true,
- `runtime-operation-inventory` reports no unsupported operation names,
- generated forward and generated derived-mask train-step representatives run
  through `mlir-runner`,
- `direct_mnist_external_runtime` remains false because the full
  `2x784 -> 8 -> 10` classifier-shaped train-step artifact has not executed
  externally.

## Goal

Make the direct MNIST runtime claim real.

The concrete target is an external runtime checksum for the full
`mnist-train-step-derived-mask` semantics over the existing fixed batch,
parameters, and labels, compared against the existing oracle family, and guarded
by the default e2e gate.

## Strategy

1. Measure artifact size, compile time, runner time, and checksum tolerances for
   exact-shape scalarized runtime generation before adding huge goldens.
2. Factor runtime generation around reusable tensor indexing and checksum
   helpers so full forward/loss/train-step cases are generated, not handwritten.
3. Land exact-shape forward runtime first.
4. Land exact-shape loss and logits checksum next.
5. Land exact-shape gradient and SGD checksum last.
6. Flip `direct_mnist_external_runtime` only if the full train-step runtime
   artifact executes externally in the default gate.
7. Close with a readiness report that records the decision and the next
   performance/compiler gap.

## Non-Goals

- Do not flip the direct runtime flag for another scaled representative.
- Do not optimize runtime performance before the full checksum exists.
- Do not depend on local MNIST downloads for runtime cases.
- Do not replace the host-side training smoke while the runtime backend is being
  scaled.
- Do not switch to IREE or StableHLO runtime packaging unless the LLVM route
  becomes impractical and the blocker is documented.

## Ticket Wave

1. `TICKET-0066`: Full Runtime Scaling Budget And Gate Plan. Completed with
   the checked `runtime-scaling-budget` report.
2. `TICKET-0067`: Runtime Tensor Indexing Codegen Helpers. Completed by
   routing `generated-dense-runtime` through reusable `RuntimeTensor` helpers
   while preserving its golden output.
3. `TICKET-0068`: Exact-Shape MNIST Forward Runtime Checksum.
4. `TICKET-0069`: Exact-Shape MNIST Loss Runtime Checksum.
5. `TICKET-0070`: Exact-Shape Derived-Mask Gradient Runtime Checksum.
6. `TICKET-0071`: Exact-Shape Derived-Mask Train-Step Runtime Checksum.
7. `TICKET-0072`: Direct Runtime Readiness Report V7.

## Exit Gate

The wave is complete when the default e2e gate either:

1. executes the exact-shape derived-mask train-step checksum externally and
   flips `direct_mnist_external_runtime`, or
2. records a precise, reproduced blocker showing why the LLVM route cannot carry
   the full artifact yet.

Anything smaller than the exact-shape train-step can be useful progress, but it
does not close the direct MNIST runtime gap.
