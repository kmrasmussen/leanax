# TICKET-0056: Runtime Scalar Math Fixture

## Problem

The existing LLVM runtime fixtures cover arithmetic, dense checksums, compare,
and select, but the classifier train step also needs scalar math for softmax and
cross entropy.

## Goal

Add a focused LLVM runtime fixture for the scalar math that blocks a direct
train-step runtime path.

## In Scope

- Exercise the LLVM/MLIR route for exponential, log, divide, and reductions or
  the closest available primitive sequence in the current shell.
- Compare the runtime result against a deterministic Python oracle.
- Keep the fixture small enough for the default e2e gate.
- Document any unsupported scalar operation that still blocks direct runtime.

## E2E Focus

Add a `runtime` manifest case for the new scalar math fixture, or a documented
expected diagnostic if the current LLVM route cannot support the operation yet.

## Acceptance Criteria

1. The fixture either executes under `mlir-runner` or fails with a precise
   documented blocker.
2. The expected scalar checksum is checked by the Rust e2e runner.
3. The runtime boundary document is updated with the result.
4. The full Nix e2e gate passes.

## First Slice

Try the smallest standalone softmax-loss checksum before touching classifier
parameter update logic.

## Status

Completed. `softmax-loss-runtime` emits an LLVM-dialect softmax cross-entropy
checksum using `llvm.intr.exp`, `llvm.intr.log`, and `llvm.fdiv`, runs through
`mlir-runner`, and compares against `0.31326166` in the Rust e2e runtime gate.
The runtime operation inventory now treats divide, exponential, and log as
covered by runtime fixture evidence.
