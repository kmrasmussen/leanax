# Runtime Wave Roadmap

The roadmap now matches the current runtime frontier after `TICKET-0058`.

The project already has generated classifier artifacts, parser checks, Python
numeric oracles, cached IDX training, structured metrics, scalar LLVM runtime
fixtures, and a tiny derived-mask train-step runtime checksum. The remaining
claim is narrower: generated classifier-shaped runtime execution is still not
done, so `direct_mnist_external_runtime` must stay false.

The new active wave is `TICKET-0059` through `TICKET-0065`. It first establishes
a shared LLVM runtime codegen skeleton and ABI, then adds generated shape-op,
reduce, and dot/dense fixtures. Only after that does it attempt generated MNIST
forward and derived-mask train-step runtime checks, followed by readiness report
v6.
