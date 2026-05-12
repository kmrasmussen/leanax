# Runtime Reduce Fixtures

`TICKET-0061` adds helper-generated reduction evidence to the runtime wave.

The new `reduce-row-runtime`, `reduce-all-runtime`, and
`reduce-keepdim-runtime` cases scalarize row-wise sum reduction, all-elements
sum reduction, and the keepdim-style rebroadcast pattern used around
softmax/loss lowering. The default e2e manifest runs all three through
`mlir-runner` and checks `36.0`, `21.0`, and `261.0`.

The runtime operation inventory now treats `stablehlo.reduce` as covered. It no
longer reports unsupported operation names for the derived-mask train-step
surface, although direct MNIST runtime remains false until the generated
classifier-shaped train-step semantics execute externally.
