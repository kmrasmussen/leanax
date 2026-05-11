# TICKET-0027: Batched Ten-Class Cross Entropy

## Problem

The current `cross-entropy-loss` artifact is a fixed two-class, single-example
slice. MNIST needs a batched ten-class loss with explicit mean semantics.

## Goal

Add a LeanAX loss artifact for batched ten-class classification and check it
against deterministic MNIST-shaped fixture values.

## In Scope

- Logits shaped `batch x 10`.
- One-hot labels shaped `batch x 10`.
- Mean cross entropy over the batch.
- Validation failures for class-count, batch-size, and rank mismatches.
- Numeric oracle coverage for deterministic fixture logits and labels.

## E2E Focus

Add `mnist-cross-entropy` as a numeric manifest case with golden text, MLIR
parsing, lowering manifest validation, and Python oracle comparison.

## Acceptance Criteria

1. LeanAX can lower a batched ten-class cross-entropy loss artifact.
2. The loss returns a scalar mean over the batch.
3. Shape/rank/class-count mismatch cases fail as expected validation failures.
4. The numeric oracle compares the generated loss against deterministic fixture
   values.
5. The full Nix e2e gate passes.

## First Slice

Use batch size `2` and class count `10`, matching the current MNIST fixture
contract.

## Status

Completed. LeanAX now has a batched ten-class `mnist-cross-entropy` artifact
with row-wise softmax denominators, scalar mean loss over the fixture batch, and
deterministic numeric oracle coverage. The IR gained a checked
`reduceSumLastDim` operation that keeps a singleton class axis, allowing row-wise
normalization to broadcast back to `2x10` logits. The e2e manifest covers
`mnist-cross-entropy` as a numeric case and
`bad-mnist-cross-entropy-shape` as an expected validation failure.
