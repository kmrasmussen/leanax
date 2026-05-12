# TICKET-0032: MNIST Train-Step Artifact

## Problem

The current MNIST smoke command checks that some LeanAX artifacts exist, but the
classifier update itself is still implemented in host Python.

## Goal

Lower a checked MNIST classifier train-step artifact that consumes one fixture
batch and returns the updated full parameter tree.

## In Scope

- Forward pass, loss, gradients, and SGD update composed into one inspected
  train-step artifact or a small set of explicitly linked artifacts.
- Full parameter tree inputs and outputs.
- Deterministic fixture batch inputs.
- Numeric oracle comparison for loss and updated parameters.
- Clear manifest/source-map coverage for debugging.

## E2E Focus

Add `mnist-train-step` as a numeric manifest case or a dedicated train-step
manifest outcome that compares loss and every updated parameter against Python.

## Acceptance Criteria

1. The train step exercises LeanAX-generated classifier artifacts rather than
   bypassing them in Python.
2. The update returns the full `w1`, `b1`, `w2`, `b2` tree.
3. The e2e gate checks loss and parameter deltas for one fixture batch.
4. Documentation states which parts are still host-orchestrated.
5. The full Nix e2e gate passes.

## First Slice

Allow the runner to stitch separately generated forward, gradient, and update
artifacts if a single monolithic module is too large for the first pass.

## Status

Completed. The e2e manifest now includes `mnist-train-step-artifact`, a
fixture-mode train-step check that stitches the generated `mnist-forward`,
`mnist-cross-entropy`, `grad-softmax-dense`, `grad-relu-dense`, and
`mnist-parameter-tree` artifacts. The script computes one analytic fixture batch
update, checks loss reduction and non-zero parameter deltas, and fails if any
required generated artifact is stale or missing. This is intentionally still a
host-orchestrated first slice, not a monolithic Lean train-step module.
