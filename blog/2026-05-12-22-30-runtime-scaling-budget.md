# Runtime Scaling Budget

`TICKET-0066` starts the exact-shape direct runtime wave with a checked budget
artifact.

The new `runtime-scaling-budget` data-loader verifies the fixed classifier
contract from the generated derived-mask train-step manifest, estimates
scalarized runtime size for exact-shape forward, loss, gradient, and train-step
cases, and writes `generated/runtime-scaling-budget.json`. The report records
the default-gate policy and the fallback if scalarized LLVM proves too large.

`mnist-progress-report` now checks that budget artifact, so the next runtime
implementation tickets have a concrete scaling decision instead of an implicit
assumption.
