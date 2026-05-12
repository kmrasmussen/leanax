# Generated Dense Runtime

`TICKET-0062` adds the generated dense bridge for the runtime wave.

The new `generated-dense-runtime` case uses the shared runtime skeleton for a
tiny `2x2 @ 2x2 + bias` composition. It returns a weighted checksum across the
four dense outputs, so row/column indexing and bias addition drift are visible
in one scalar.

The default e2e manifest emits the fixture, compares it to golden LLVM MLIR,
runs it through `mlir-runner`, and checks `15.25`. This is still not direct
MNIST runtime execution; it is the generated dense building block before the
forward and train-step runtime checksums.
