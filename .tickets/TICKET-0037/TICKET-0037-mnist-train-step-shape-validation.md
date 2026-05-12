# TICKET-0037: MNIST Train-Step Shape Validation Suite

## Problem

The full classifier train step has many shape boundaries. Without explicit
negative cases, a future monolithic artifact could accept mismatched batches,
labels, hidden dimensions, or parameter-tree shapes and fail later in less
obvious places.

## Goal

Add expected validation failures that pin the shape contract for the monolithic
MNIST train-step artifact.

## In Scope

- Batch-size mismatch between images, labels, and intermediate gradients.
- Class-count mismatch between logits, labels, `w2`, and `b2`.
- Hidden-width mismatch between `w1`, `b1`, `w2`, and ReLU gradient state.
- Parameter-tree mismatch for all four returned update tensors.
- Clear stderr snippets that identify the failing boundary.

## E2E Focus

Add `bad-mnist-train-step-*` expected validation-fail cases to the manifest.

## Acceptance Criteria

1. Each train-step shape family has at least one expected validation failure.
2. The failures do not emit stale generated output files.
3. Error messages identify the operation or parameter boundary that failed.
4. The validation-fail count in the runner summary increases intentionally.
5. The full Nix e2e gate passes.

## First Slice

Start with one hidden-width mismatch and one label/logit class-count mismatch.

## Status

Completed. The manifest now includes `bad-mnist-train-step-label-shape`,
`bad-mnist-train-step-hidden-shape`, and
`bad-mnist-train-step-parameter-shape` as expected validation failures. These
cover class-count, explicit ReLU mask, and parameter-update boundaries for the
monolithic train-step contract.
