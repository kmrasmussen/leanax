# Runtime Capability Matrix

`TICKET-0041` makes the runtime boundary explicit again.

The new `runtime-capability-matrix` manifest case asserts the tools the current
gate actually needs: `mlir-opt` for parsing and `mlir-runner` for the LLVM
runtime fixture. It also reports optional direct StableHLO/IREE tooling without
turning absent tools into a false claim of runtime support.

This keeps the next runtime tickets grounded: dense and MNIST-forward runtime
slices can build on the LLVM path while direct StableHLO runtime remains a
reported capability gap until the shell really provides it.
