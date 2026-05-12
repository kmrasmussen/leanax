# TICKET-0071: Exact-Shape Derived-Mask Train-Step Runtime Checksum

## Problem

`direct_mnist_external_runtime` cannot flip until the exact-shape
`mnist-train-step-derived-mask` semantics execute externally and compare loss
plus updated parameter evidence against an oracle.

## Goal

Generate and execute the exact-shape derived-mask train-step runtime checksum.

## In Scope

- Use the full `2x784 -> 8 -> 10` classifier-shaped train-step contract.
- Compute forward pass, softmax loss, derived ReLU mask, gradients, and one SGD
  update.
- Return checksum evidence over loss and updated `w1`, `b1`, `w2`, and `b2`.
- Compare against the oracle used by host-side train-step checks.
- Decide whether to flip `direct_mnist_external_runtime`.

## E2E Focus

Add a default-gate runtime case if it fits the budget, or an opt-in gate plus a
readiness blocker if it does not.

## Acceptance Criteria

1. Exact-shape train-step runtime evidence exists.
2. The checksum covers loss and all updated parameters.
3. The readiness report explains whether the direct runtime flag flips.
4. The full Nix e2e gate passes.

## First Slice

Build on `TICKET-0070`; add SGD updates and final checksums without changing the
training data source.

## Status

Ready for analysis.
