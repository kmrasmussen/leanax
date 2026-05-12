# TICKET-0057: Tiny Derived-Mask Train-Step Runtime Fixture

## Problem

`mnist-forward-runtime` proves only a small forward checksum, not the derived
mask train-step semantics that update parameters.

## Goal

Add a tiny runtime fixture that mirrors the derived-mask train-step structure on
small tensors before scaling toward the full `2x784 -> 8 -> 10` artifact.

## In Scope

- Use a tiny batch, hidden size, and class count.
- Include derived ReLU mask logic, softmax-style loss, gradients, and SGD
  updates when the scalar math fixture supports them.
- Return one or more scalar checksums that prove updated parameters and loss.
- Compare against the existing Python oracle style.

## E2E Focus

Add a manifested runtime case once the scalar math route is proven.

## Acceptance Criteria

1. The runtime fixture derives the mask internally.
2. The fixture returns loss and parameter-update evidence, not only logits.
3. Runtime checks compare against deterministic oracle values.
4. The full Nix e2e gate passes.

## First Slice

Start with a tiny shape that exercises every train-step operation category
without producing a huge LLVM fixture.

## Status

Completed. `tiny-train-step-runtime` is a scalar-expanded LLVM runtime fixture
with a tiny batch, two inputs, two hidden units, and two classes. It derives the
ReLU mask internally, computes softmax loss, backpropagates through both dense
layers, applies one SGD update, and returns a checksum covering loss plus all
updated parameters. The Rust runtime gate compares the `mlir-runner` result
against `7.939712`.
