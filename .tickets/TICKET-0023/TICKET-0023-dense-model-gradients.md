# TICKET-0023: Dense Model Gradients

## Problem

`grad-square-sum` proves the gradient pipeline for a narrow scalar expression,
but MNIST training needs gradients through dense layers and a loss.

## Goal

Generate gradients for a tiny dense-model loss and check them numerically.

## In Scope

- Derivative rules for matmul, add, broadcast, ReLU, and the selected loss
  subset.
- Adjoint accumulation for repeated parameter use if required by the model.
- Tiny finite-difference or Python analytic oracle.
- Explicit limitations for unsupported control flow or dynamic shapes.

## E2E Focus

Add a `grad-dense-loss` numeric case comparing generated gradients against a
Python oracle on a tiny fixed-shape model.

## Acceptance Criteria

1. LeanAX can generate a gradient module for a scalar dense-model loss.
2. The generated gradients match the oracle within documented tolerance.
3. Unsupported gradient cases fail with structured validation errors.
4. Existing `grad-square-sum` coverage continues to pass.
5. The full Nix e2e gate passes.

## First Slice

Target a one-layer dense model before expanding to the full two-layer MLP.

## Status

Completed. `LeanAX/Grad.lean` now emits `grad-dense-loss`, a one-layer dense
gradient artifact for `sum((x @ w + b)^2)` returning `grad_w`. The e2e oracle
checks the generated matrix against the analytic Python result, and
`bad-grad-dense-shape` covers an unsupported gradient shape as an expected
validation failure.
