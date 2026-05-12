# Generated Forward Runtime

`TICKET-0063` adds a generated classifier-forward runtime representative.

The new `generated-mnist-forward-runtime` case uses the shared runtime skeleton
for a dense-ReLU-dense path with ReLU compare/select and a weighted logits
checksum. The default e2e manifest emits the fixture, compares it to golden LLVM
MLIR, runs it through `mlir-runner`, and checks `-0.525`.

This is intentionally documented as a scaled representative, not the full
`2x784 -> 8 -> 10` classifier artifact. It proves the generated forward
composition path before the next ticket attempts generated derived-mask
train-step runtime execution.
