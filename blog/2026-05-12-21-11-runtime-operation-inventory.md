# Runtime Operation Inventory

`TICKET-0055` adds a manifest-backed runtime operation inventory for the
derived-mask MNIST train step.

The verifier reads the lowering manifest instead of scraping MLIR text. It
checks the fixed train-step input and output contract, verifies that the
operation set is still the expected thirteen StableHLO-shaped operations, and
prints stable operation counts.

The current unsupported runtime expansion surface is now explicit in the e2e
output: broadcast, divide, exponential, log, reduce, reshape, and transpose.
That gives the scalar math/runtime fixture tickets a concrete target instead of
a vague "make runtime work" milestone.
