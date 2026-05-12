# LeanAX Tickets

This backlog tracks durable project work for LeanAX.

Use tickets for implementation slices that should survive across coding
sessions: IR design, lowering, proof targets, e2e harness work, and runtime
integration. Avoid tickets for every typo or tiny fixture adjustment.

Every implementation ticket should point at an observable e2e capability.

## Active Wave

The current ready-for-analysis wave is `TICKET-0059` through `TICKET-0065`.
It moves runtime work from handwritten scalar fixtures toward generated
classifier-shaped runtime checks:

1. runtime LLVM codegen skeleton and ABI,
2. shape-op runtime fixtures,
3. reduce runtime fixtures,
4. dot/dense runtime fixture,
5. generated MNIST forward runtime checksum,
6. generated derived-mask train-step runtime checksum,
7. runtime readiness report v6.
