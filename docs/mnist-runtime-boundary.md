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

## Next Ticket Queue

1. `TICKET-0055`: Runtime Operation Inventory Verifier.
2. `TICKET-0056`: Runtime Scalar Math Fixture.
3. `TICKET-0057`: Tiny Derived-Mask Train-Step Runtime Fixture.
4. `TICKET-0058`: Runtime Readiness Report V5.

## Current Progress

`runtime-operation-inventory` now verifies the
`mnist-train-step-derived-mask` lowering manifest during the default e2e gate.
It checks the fixed input/output contract, the expected thirteen-operation
surface, and the current runtime expansion gap:

- `stablehlo.broadcast_in_dim`
- `stablehlo.divide`
- `stablehlo.exponential`
- `stablehlo.log`
- `stablehlo.reduce`
- `stablehlo.reshape`
- `stablehlo.transpose`

Those operations are the next concrete bridge between the existing handwritten
LLVM runtime fixtures and a direct runtime train-step artifact.
