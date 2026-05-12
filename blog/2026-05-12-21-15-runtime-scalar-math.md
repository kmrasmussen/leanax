# Runtime Scalar Math

`TICKET-0056` adds the first runtime fixture for softmax-style scalar math.

The new `softmax-loss-runtime` case emits LLVM-dialect MLIR with
`llvm.intr.exp`, `llvm.intr.log`, and `llvm.fdiv`. It computes the
cross-entropy loss for logits `[1.0, 2.0]` with class `1`, and the Rust e2e
runtime gate checks the `mlir-runner` result against `0.31326166`.

This does not run the full classifier train step yet, but it removes the
highest-risk scalar math blocker from the runtime inventory. The remaining
runtime expansion surface is now broadcast, reduce, reshape, and transpose.
