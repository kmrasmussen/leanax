# TICKET-0069: Exact-Shape MNIST Loss Runtime Checksum

## Problem

The exact-shape forward checksum will not by itself prove the softmax
cross-entropy path for the classifier-sized logits and labels.

## Goal

Generate and execute an exact-shape MNIST forward-plus-loss runtime checksum.

## In Scope

- Reuse the exact-shape forward runtime path from `TICKET-0068`.
- Compute the row-wise softmax denominator, probabilities, selected log
  probabilities, and mean loss for `2x10` logits and labels.
- Return loss plus an optional logits checksum.
- Compare against the existing Python oracle family.
- Keep tolerance explicit.

## E2E Focus

Manifest an exact-shape loss runtime case through `mlir-runner` if the budget
allows; otherwise record the reproduced budget blocker.

## Acceptance Criteria

1. Exact-shape loss runtime evidence exists.
2. Runtime loss agrees with the oracle within explicit tolerance.
3. The case is separate from full gradient/train-step runtime.
4. The full Nix e2e gate passes.

## First Slice

Start with forward-plus-loss only. Do not update parameters in this ticket.

## Status

Completed. `exact-mnist-loss-runtime` reuses the exact forward body, adds fixed
`2x10` labels, computes row-wise softmax cross-entropy, and returns mean loss
through `mlir-runner`. `exact-mnist-loss-runtime-oracle` checks the manifest
expectation against the deterministic Python oracle.
