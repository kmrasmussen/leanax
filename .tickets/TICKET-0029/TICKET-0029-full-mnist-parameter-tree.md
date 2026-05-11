# TICKET-0029: Full MNIST Parameter Tree

## Problem

The current parameter-tree update covers one weight matrix and one bias vector.
A classifier train step needs `w1`, `b1`, `w2`, and `b2` updated together.

## Goal

Represent and lower a full two-layer MNIST classifier parameter tree update.

## In Scope

- Parameters `w1`, `b1`, `w2`, and `b2`.
- Matching gradient tensors for every parameter.
- One SGD learning-rate scalar.
- Multi-output module returning all updated parameters.
- Numeric oracle coverage for every returned tensor.

## E2E Focus

Add `mnist-parameter-tree` as a numeric manifest case and compare all four
updated outputs against a Python oracle.

## Acceptance Criteria

1. LeanAX can name and validate the full classifier parameter tree.
2. The generated update returns `next_w1`, `next_b1`, `next_w2`, and `next_b2`.
3. Shape mismatch validation covers at least one full-tree failure mode.
4. The numeric oracle checks every updated tensor.
5. The full Nix e2e gate passes.

## First Slice

Reuse the existing two-parameter update pattern, then expand it to four outputs.
