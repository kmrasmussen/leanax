# TICKET-0062: Runtime Dot/Dense Lowering Fixture

## Problem

Dense-layer runtime evidence exists, but it is not yet produced from the shared
generated runtime lowering path. The MNIST classifier route needs dot/dense
lowering that can be reused by generated forward and train-step checks.

## Goal

Add a generated dot/dense runtime fixture using the runtime skeleton and compare
it against an oracle checksum.

## In Scope

- Generate a fixed-shape dense or dot runtime case from the lowering metadata
  path rather than a one-off handwritten fixture.
- Include bias add if that is the smallest useful bridge to classifier forward.
- Compare against deterministic oracle values.
- Keep artifact naming and manifest entries consistent with earlier runtime
  cases.

## E2E Focus

Run the generated dot/dense fixture through `mlir-runner` in the default e2e
gate.

## Acceptance Criteria

1. A generated dot/dense runtime fixture executes externally.
2. The fixture uses the shared runtime skeleton from this wave.
3. The checksum agrees with the oracle.
4. The full Nix e2e gate passes.

## First Slice

Use a tiny dense layer with fixed constants, then ensure the same machinery can
be called by the MNIST forward runtime ticket.

## Status

Ready for analysis.
