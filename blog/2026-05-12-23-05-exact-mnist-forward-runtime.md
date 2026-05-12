# Exact MNIST Forward Runtime

`TICKET-0068` adds the first full-size classifier runtime artifact.

`exact-mnist-forward-runtime` emits the fixed `2x784 -> 8 -> 10`
dense-ReLU-dense forward pass through the LLVM runtime path. The generated
golden is 33,443 lines, executes with `mlir-runner`, and returns the logits
checksum `3.970676`.

A new `exact-mnist-forward-runtime-oracle` data-loader computes the same
checksum from the deterministic Python oracle tensors and checks the manifest
expectation. This proves the forward scale-up without claiming the full
train-step is externally executable yet.
