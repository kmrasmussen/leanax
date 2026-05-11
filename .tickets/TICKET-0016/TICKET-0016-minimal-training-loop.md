# TICKET-0016: Minimal Host-Side Training Loop

## Goal

Add enough host-side orchestration to run a tiny trainable model on synthetic
data before moving to MNIST.

## E2E Focus

The gate should assert that loss decreases on a small deterministic dataset.

## Status

Completed. `LeanAX/Training.lean` adds a checked scalar train-step update module,
and `e2e/python/training_loop.py` asserts loss decreases on deterministic
synthetic linear data in the unified e2e gate.
