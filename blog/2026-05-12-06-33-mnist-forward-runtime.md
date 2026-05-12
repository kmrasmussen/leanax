# MNIST Forward Runtime Slice

`TICKET-0043` adds the first classifier-forward-shaped runtime fixture.

The new `mnist-forward-runtime` case is intentionally a stepping-stone shape:
a small dense-ReLU-dense computation reduced to a scalar checksum. It executes
through LLVM `mlir-runner` and checks the result `66.6125`.

This keeps the runtime claim precise. LeanAX now has forward-pattern runtime
coverage beyond a dense checksum, but the full `2x784 -> 8 -> 10` MNIST forward
artifact still needs a stronger runtime path before it can be claimed as direct
classifier runtime execution.
