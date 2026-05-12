# TICKET-0068: Exact-Shape MNIST Forward Runtime Checksum

## Problem

`generated-mnist-forward-runtime` proves a scaled dense-ReLU-dense
representative, not the full `2x784 -> 8 -> 10` classifier forward shape.

## Goal

Generate and execute an exact-shape MNIST classifier forward runtime checksum
through the external LLVM runtime path.

## In Scope

- Use the existing fixed batch, parameters, and deterministic fixture values.
- Generate the full `2x784 -> 8 -> 10` dense-ReLU-dense forward computation.
- Return a checksum over logits or logits plus hidden activations.
- Compare against the existing Python oracle family.
- Keep the case in the default gate only if it fits the budget from
  `TICKET-0066`.

## E2E Focus

Manifest an exact-shape runtime case and run it through `mlir-runner`, or record
a reproduced budget blocker if it is too large for the default gate.

## Acceptance Criteria

1. Exact-shape forward runtime evidence exists.
2. The checksum agrees with a deterministic oracle.
3. The readiness report distinguishes this from full train-step runtime.
4. The full Nix e2e gate passes.

## First Slice

Generate only forward logits and one checksum. Do not include loss or gradients
in this ticket.

## Status

Ready for analysis.
