# TICKET-0033: Ten-Class MNIST Smoke Command

## Problem

The current MNIST smoke command trains a tiny parity classifier over two
features. It is useful, but it is not a ten-class MNIST classifier smoke.

## Goal

Replace or extend the smoke command with a ten-class fixture-mode classifier run
that uses the checked MNIST-shaped artifacts.

## In Scope

- Fixture-mode ten-class labels.
- MNIST-shaped image vectors.
- Metrics for loss and accuracy.
- Checks that the generated MNIST forward, loss, gradient, and update artifacts
  are present and current.
- Stable e2e thresholds that avoid flaky metric assertions.

## E2E Focus

Add a training-loop case for the ten-class MNIST classifier smoke and assert
loss improves or accuracy does not regress over a deterministic short run.

## Acceptance Criteria

1. A documented command runs ten-class fixture-mode MNIST classifier training.
2. The command exercises the checked classifier artifacts from the previous
   tickets.
3. The output prints first/final loss and accuracy.
4. The e2e gate fails on metric regression or stale generated artifacts.
5. The full Nix e2e gate passes.

## First Slice

Run over the deterministic fixture only; full-dataset support is a separate
ticket.

## Status

Completed. `e2e/python/mnist_classifier_smoke.py` is the ten-class fixture-mode
classifier smoke command. It trains the `784 -> 8 -> 10` classifier over the
deterministic MNIST-shaped fixture, requires the generated forward, loss,
gradient, and full parameter-tree artifacts, prints first/final loss and
accuracy, and fails if loss does not improve or accuracy regresses. The e2e
manifest covers it as a `training-loop` case.
