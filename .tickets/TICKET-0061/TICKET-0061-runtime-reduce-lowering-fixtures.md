# TICKET-0061: Runtime Reduce Lowering Fixtures

## Problem

The train-step path depends on reductions for row-wise softmax, loss
aggregation, gradient accumulation, and checksum production. The current runtime
fixtures exercise scalar math and tiny train-step logic, but reduce lowering is
not yet proven as a generated reusable operation.

## Goal

Add generated LLVM runtime fixtures for fixed-shape reductions that match the
classifier train-step needs.

## In Scope

- Cover row-wise reductions used by softmax-style normalization.
- Cover all-elements reductions used by scalar loss and checksums.
- Cover the reshape/keepdim-style patterns needed around reductions when they
  occur in the generated artifact.
- Compare results against deterministic Python oracle values.
- Keep numerical tolerance explicit.

## E2E Focus

Add manifested runtime cases that execute generated reduce lowering through
`mlir-runner`.

## Acceptance Criteria

1. Row-wise and all-elements reduce fixtures execute externally.
2. Expected values are generated or checked by the existing oracle style.
3. The fixtures are reusable by later forward and train-step runtime tickets.
4. The full Nix e2e gate passes.

## First Slice

Start with sum reductions over tiny tensors, then add the minimum extra pattern
needed by the softmax/loss path.

## Status

Completed. `reduce-row-runtime`, `reduce-all-runtime`, and
`reduce-keepdim-runtime` now use the shared runtime skeleton to cover row-wise
sum reduction, all-elements sum reduction, and the keepdim-style rebroadcast
pattern needed around softmax/loss lowering. The default manifest executes them
through `mlir-runner` with deterministic checksums `36.0`, `21.0`, and `261.0`.
The runtime operation inventory now includes `stablehlo.reduce` in the covered
fixture surface and reports no unsupported operation names.
