# TICKET-0063: Generated MNIST Forward Runtime Checksum

## Problem

`mnist-forward-runtime` proves a small forward checksum, but the next frontier is
a generated runtime checksum for the classifier forward path. Without that
bridge, the project still cannot say generated classifier artifacts execute
externally.

## Goal

Generate an external runtime checksum for the MNIST classifier forward semantics
using the runtime lowering machinery from the earlier wave tickets.

## In Scope

- Reuse generated lowering metadata and runtime skeletons.
- Include dense, bias, ReLU/mask-compatible compare/select behavior if needed,
  and shape ops required by the classifier forward path.
- Compare logits or a deterministic logits checksum against the existing oracle
  style.
- Keep the fixture fixed-shape and small enough for normal e2e execution.
- Document whether the case is full classifier-shaped or a scaled
  representative, and why.

## E2E Focus

Add a default-gate runtime case whose expected value is derived from the same
numeric oracle family as the generated MNIST artifacts.

## Acceptance Criteria

1. A generated MNIST forward runtime checksum executes externally.
2. The runtime checksum agrees with a deterministic oracle.
3. The report distinguishes forward runtime coverage from train-step runtime
   coverage.
4. The full Nix e2e gate passes.

## First Slice

Target the classifier-shaped forward path if practical. If artifact size blocks
that, land the largest honest scaled representative and document the exact
remaining gap.

## Status

Ready for analysis.
