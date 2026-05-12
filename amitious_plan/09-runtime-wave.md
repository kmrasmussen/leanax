# Runtime Wave Roadmap

This wave starts from the current state after `TICKET-0058`:

- generated MNIST classifier artifacts are covered by golden text, MLIR parsing,
  lowering manifests, and Python numeric oracles,
- cached IDX training and structured metrics pass in the default e2e gate,
- LLVM `mlir-runner` executes scalar/runtime fixtures including a tiny
  derived-mask train-step checksum,
- `direct_mnist_external_runtime` is still false because the full
  classifier-shaped train-step artifact has not executed externally.

## Goal

Move from handwritten scalar runtime fixtures to generated runtime lowering for
the classifier-shaped path.

The next honest target is not a general tensor compiler. It is a fixed-shape,
auditable runtime bridge for the operation surface used by
`mnist-train-step-derived-mask`.

## Strategy

1. Define a scalarized runtime ABI and codegen skeleton.
2. Prove the remaining shape-only tensor ops: broadcast, reshape, and transpose.
3. Prove reduce lowering for row-wise and all-elements reductions.
4. Prove generated dot/dense lowering instead of handwritten dense fixtures.
5. Generate a classifier-forward runtime checksum from the same style of
   lowering path.
6. Generate a derived-mask train-step runtime checksum and only then consider
   flipping `direct_mnist_external_runtime`.

## Non-Goals

- Do not package IREE or StableHLO as part of this wave.
- Do not build a general dynamic-shape tensor runtime.
- Do not make the runtime fixture depend on local MNIST downloads.
- Do not mark direct full MNIST runtime true for tiny scalar fixtures.

## Ticket Wave

1. `TICKET-0059`: Runtime LLVM Codegen Skeleton And ABI. Completed by the
   helper-generated `generated-arithmetic-runtime` case.
2. `TICKET-0060`: Runtime Shape Ops Lowering Fixtures. Completed by the
   helper-generated broadcast, reshape, and transpose runtime checksums.
3. `TICKET-0061`: Runtime Reduce Lowering Fixtures. Completed by row-wise,
   all-elements, and keepdim-style runtime checksums.
4. `TICKET-0062`: Runtime Dot/Dense Lowering Fixture.
5. `TICKET-0063`: Generated MNIST Forward Runtime Checksum.
6. `TICKET-0064`: Generated Derived-Mask Train-Step Runtime Checksum.
7. `TICKET-0065`: Runtime Readiness Report V6.

## Exit Gate

The wave is complete when the default e2e gate contains a generated external
runtime checksum for the derived-mask train-step semantics and the readiness
report explains whether that is strong enough to flip
`direct_mnist_external_runtime`.
