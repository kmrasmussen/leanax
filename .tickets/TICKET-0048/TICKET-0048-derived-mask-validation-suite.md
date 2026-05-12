# TICKET-0048: Derived Mask Validation Suite

## Problem

Comparison and select add new dtype boundaries. Without negative cases, predicate
shape mismatches and selected-value mismatches could regress silently.

## Goal

Add expected validation failures for compare/select and derived-mask train-step
shape boundaries.

## In Scope

- Compare operand shape mismatch.
- Select predicate/result shape mismatch.
- Select value dtype/shape mismatch.
- Derived train-step parameter mismatch that reaches the new mask path.

## E2E Focus

Add `bad-compare-*`, `bad-select-*`, or `bad-mnist-derived-mask-*` manifest
cases with stable stderr snippets.

## Acceptance Criteria

1. Predicate/value shape errors fail as expected validation failures.
2. Selected f32 tensors must match each other.
3. Predicate tensors must match the selected tensor shape.
4. The validation-fail count increases intentionally.
5. The full Nix e2e gate passes.

## First Slice

Start with one compare shape mismatch and one select predicate shape mismatch.

## Status

Completed. The manifest now includes `bad-compare-shape` and
`bad-select-predicate-shape` expected failures. They prove compare operands must
match and select predicates must match the selected f32 tensor shape before any
invalid MLIR is written.
