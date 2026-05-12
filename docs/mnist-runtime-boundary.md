# Direct MNIST Runtime Boundary

The next direct-runtime route for LeanAX should be LLVM lowering expansion
through the existing `mlir-runner` gate.

This is a pragmatic boundary, not the final architecture. The current Nix shell
has `mlir-opt` and `mlir-runner`, and the e2e manifest already executes
`affine-runtime`, `dense-runtime`, and `mnist-forward-runtime` through that path.
The same runtime capability matrix still reports `stablehlo-opt`,
`iree-compile`, and `iree-run-module` unavailable, so direct StableHLO or IREE
execution is not yet a reliable default-gate route for this repository.

## Decision

Choose LLVM lowering expansion for the next slice.

Do not make PyPI IREE, an unpackaged StableHLO toolchain, or ad hoc host FHS
wrapping the next default path. Those routes may become attractive later, but
they currently add packaging risk before the project has a precise operation
inventory and scalar runtime fixtures for the classifier math.

## Current Runtime Evidence

The current runtime gate proves:

- `runtime-capability-matrix`: required `mlir-opt` and `mlir-runner` are
  available; optional `stablehlo-opt`, `iree-compile`, and `iree-run-module`
  are unavailable.
- `affine-runtime`: LLVM-dialect scalar arithmetic executes through
  `mlir-runner` and returns `94.25`.
- `dense-runtime`: a fixed dense-layer checksum executes through
  `mlir-runner` and returns `103.375`.
- `mnist-forward-runtime`: a small dense-ReLU-dense checksum executes through
  `mlir-runner` and returns `66.6125`.
- `broadcast-shape-runtime`, `reshape-shape-runtime`, and
  `transpose-shape-runtime`: helper-generated fixed-shape indexing checksums
  execute through `mlir-runner` and return `46.0`, `91.0`, and `86.0`.
- `reduce-row-runtime`, `reduce-all-runtime`, and `reduce-keepdim-runtime`:
  helper-generated reduction checksums execute through `mlir-runner` and return
  `36.0`, `21.0`, and `261.0`.
- `generated-dense-runtime`: a helper-generated dense composition executes
  through `mlir-runner` and returns `15.25`.
- `generated-mnist-forward-runtime`: a helper-generated dense-ReLU-dense
  classifier-forward representative executes through `mlir-runner` and returns
  `-0.525`.
- `exact-mnist-forward-runtime`: the full fixed `2x784 -> 8 -> 10`
  dense-ReLU-dense forward computation executes through `mlir-runner` and
  returns `3.970676`, with a Python oracle checking the expected logits
  checksum.
- `exact-mnist-loss-runtime`: the same exact forward body plus fixed `2x10`
  labels executes row-wise softmax cross-entropy through `mlir-runner` and
  returns mean loss `2.261078`, with a Python oracle checking the expected loss.
- `generated-derived-mask-train-step-runtime`: a helper-generated derived-mask
  train-step representative executes through `mlir-runner` and returns
  `1.4609127`.

The current classifier artifact proves the compiler-side shape:

- input batch: `x : tensor<2x784xf32>`
- labels: `labels : tensor<2x10xf32>`
- parameters: `w1 : tensor<784x8xf32>`, `b1 : tensor<8xf32>`,
  `w2 : tensor<8x10xf32>`, `b2 : tensor<10xf32>`
- outputs: `next_w1 : tensor<784x8xf32>`, `next_b1 : tensor<8xf32>`,
  `next_w2 : tensor<8x10xf32>`, `next_b2 : tensor<10xf32>`, and
  `loss : tensor<f32>`

## Required Operation Surface

`mnist-train-step-derived-mask` currently needs these StableHLO-shaped
operations:

- `stablehlo.constant`
- `stablehlo.broadcast_in_dim`
- `stablehlo.add`
- `stablehlo.multiply`
- `stablehlo.divide`
- `stablehlo.exponential`
- `stablehlo.log`
- `stablehlo.compare`
- `stablehlo.select`
- `stablehlo.transpose`
- `stablehlo.reshape`
- `stablehlo.reduce`
- `stablehlo.dot_general`

The important tensor shapes are the classifier contract above plus intermediate
`2x8`, `2x10`, `2x1`, `8x2`, `10x2`, `784x2`, and `10x8` tensors.

## What Counts As Direct Full MNIST Runtime Execution

Direct full MNIST runtime execution requires an external runtime path to consume
the classifier train-step semantics, not just Python reference math. The minimum
credible milestone is:

1. emit or lower the `mnist-train-step-derived-mask` computation into an
   executable external-runtime artifact,
2. run that artifact under a reproducible tool in the Nix shell,
3. compare `loss` and updated parameter checksums against the existing Python
   oracle for the same fixed batch and parameters,
4. include the check in the default e2e manifest, and
5. update `mnist-progress-report` so `direct_mnist_external_runtime` only flips
   when that manifested runtime check passes.

Returning only a small forward checksum is not enough. Running host-side Python
training over generated text is not enough. Parsing StableHLO-shaped text with
`mlir-opt --allow-unregistered-dialect` is not enough.

## Completed Runtime Ticket Queue

1. `TICKET-0055`: Runtime Operation Inventory Verifier.
2. `TICKET-0056`: Runtime Scalar Math Fixture.
3. `TICKET-0057`: Tiny Derived-Mask Train-Step Runtime Fixture.
4. `TICKET-0058`: Runtime Readiness Report V5.
5. `TICKET-0059`: Runtime LLVM Codegen Skeleton And ABI.
6. `TICKET-0060`: Runtime Shape Ops Lowering Fixtures.
7. `TICKET-0061`: Runtime Reduce Lowering Fixtures.
8. `TICKET-0062`: Runtime Dot/Dense Lowering Fixture.
9. `TICKET-0063`: Generated MNIST Forward Runtime Checksum.
10. `TICKET-0064`: Generated Derived-Mask Train-Step Runtime Checksum.
11. `TICKET-0065`: Runtime Readiness Report V6.

## Next Ticket Queue

1. `TICKET-0066`: Full Runtime Scaling Budget And Gate Plan.
2. `TICKET-0067`: Runtime Tensor Indexing Codegen Helpers.
3. `TICKET-0068`: Exact-Shape MNIST Forward Runtime Checksum.
4. `TICKET-0069`: Exact-Shape MNIST Loss Runtime Checksum.
5. `TICKET-0070`: Exact-Shape Derived-Mask Gradient Runtime Checksum.
6. `TICKET-0071`: Exact-Shape Derived-Mask Train-Step Runtime Checksum.
7. `TICKET-0072`: Direct Runtime Readiness Report V7.

## Current Progress

`runtime-operation-inventory` verifies the `mnist-train-step-derived-mask`
lowering manifest during the default e2e gate. It checks the fixed input/output
contract, the expected thirteen-operation surface, and the current runtime
expansion gap.

`softmax-loss-runtime` now covers scalar softmax loss math through LLVM
intrinsics for exp/log plus floating-point division, so divide, exponential,
and log are no longer in the unsupported fixture surface.

`tiny-train-step-runtime` now covers a scalar-expanded train-step checksum with
internal ReLU mask derivation, softmax loss, gradients for both dense layers,
and SGD updates. It is intentionally tiny and scalar-expanded; it proves the
runtime can execute train-step semantics, not that the full fixed
`2x784 -> 8 -> 10` classifier artifact is already executable.

`mnist-progress-report` now distinguishes these runtime milestones from direct
full MNIST runtime execution. Runtime operation inventory, scalar math runtime,
shape-op runtime fixtures, reduce runtime fixtures, and tiny train-step runtime
are true;
`direct_mnist_external_runtime` remains false until the full classifier-shaped
train-step artifact executes externally.

The runtime operation inventory no longer reports unsupported operation names,
and generated forward/train-step representatives now execute externally. The
remaining bridge is scale: the full classifier-shaped `2x784 -> 8 -> 10`
train-step artifact still has not executed externally, so
`direct_mnist_external_runtime` remains false.

`mnist-progress-report` records this as runtime readiness v6: generated runtime
codegen, shape ops, reductions, dense composition, forward representative, and
train-step representative are all true, while direct full MNIST external runtime
is still false.

`runtime-scaling-budget` now records the planned exact-shape scalarized LLVM
budget and default-gate policy before the next wave starts generating large
runtime artifacts.

`generated-dense-runtime` now runs through the reusable `RuntimeTensor` helper
layer. That gives the exact-shape forward/loss/train-step tickets a structured
way to enumerate row-major scalar refs instead of growing more handwritten
runtime strings.

`exact-mnist-forward-runtime` and `exact-mnist-loss-runtime` now close the
forward/loss scale gap: the full fixed classifier forward pass and mean
cross-entropy run externally, while direct full MNIST runtime execution remains
false because gradients, SGD updates, and the train-step checksum still need
exact-shape runtime coverage.
