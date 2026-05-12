# LeanAX Tickets

This backlog tracks durable project work for LeanAX.

Use tickets for implementation slices that should survive across coding
sessions: IR design, lowering, proof targets, e2e harness work, and runtime
integration. Avoid tickets for every typo or tiny fixture adjustment.

Every implementation ticket should point at an observable e2e capability.

## Latest Completed Wave

`TICKET-0059` through `TICKET-0065` moved runtime work from handwritten scalar
fixtures toward generated
classifier-shaped runtime checks:

1. runtime LLVM codegen skeleton and ABI,
2. shape-op runtime fixtures,
3. reduce runtime fixtures,
4. dot/dense runtime fixture,
5. generated MNIST forward runtime checksum,
6. generated derived-mask train-step runtime checksum,
7. runtime readiness report v6.

The wave leaves `direct_mnist_external_runtime` false because the generated
forward and train-step checks are scaled representatives, not the full
`2x784 -> 8 -> 10` classifier-shaped train-step artifact.

## Active Wave

`TICKET-0066` through `TICKET-0072` target the exact-shape direct runtime gap:

1. full runtime scaling budget and gate plan,
2. tensor indexing codegen helpers,
3. exact-shape forward checksum,
4. exact-shape loss checksum,
5. exact-shape gradient checksum,
6. exact-shape train-step checksum,
7. direct runtime readiness report v7.

The active wave should not flip `direct_mnist_external_runtime` until the full
`mnist-train-step-derived-mask` semantics execute externally in the default
gate.
