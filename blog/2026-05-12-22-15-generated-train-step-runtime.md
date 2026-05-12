# Generated Train-Step Runtime

`TICKET-0064` adds a generated derived-mask train-step runtime representative.

The new `generated-derived-mask-train-step-runtime` case uses the shared runtime
skeleton for forward pass, internal ReLU mask derivation, softmax loss,
gradients, one SGD update, and a checksum over loss plus updated parameters.
The default e2e manifest emits the fixture, compares it to golden LLVM MLIR,
runs it through `mlir-runner`, and checks `1.4609127`.

This does not flip `direct_mnist_external_runtime`. The fixture is a scaled
representative, while the direct-runtime claim still requires the full
`2x784 -> 8 -> 10` classifier-shaped train-step artifact to execute externally.
