# Tiny Train-Step Runtime

`TICKET-0057` adds a tiny scalar-expanded train-step runtime fixture.

The new `tiny-train-step-runtime` case derives a ReLU mask internally, computes
softmax loss, backpropagates through two dense layers, applies SGD updates, and
returns one checksum covering the loss plus all updated parameters. The Rust
runtime gate runs the emitted LLVM-dialect MLIR through `mlir-runner` and checks
the result against `7.939712`.

This is still not direct full MNIST runtime execution. It proves the train-step
semantics on a small scalar-expanded shape before trying to lower the full
`2x784 -> 8 -> 10` classifier artifact.
