# TICKET-0047: MNIST Train Step With Derived Mask

## Problem

`mnist-train-step` is monolithic, but its API still includes `relu_mask`. That
keeps a host-side control edge in the middle of the train-step contract.

## Goal

Add a derived-mask MNIST train-step artifact that computes the ReLU mask inside
the lowered module.

## In Scope

- Inputs for images, labels, `w1`, `b1`, `w2`, and `b2`.
- Hidden pre-activation, derived ReLU activation, and derived f32 mask.
- The same softmax dense gradient, first-layer gradient, SGD update, and loss
  outputs as the explicit-mask train step.
- Numeric oracle comparison against the same deterministic classifier math.

## E2E Focus

Add `mnist-train-step-derived-mask` as a numeric manifest case and keep the
existing explicit-mask `mnist-train-step` case as a compatibility fixture.

## Acceptance Criteria

1. The derived-mask train step has no `relu_mask` input.
2. The artifact returns all four updated parameter tensors plus loss.
3. The oracle verifies the same update as the explicit-mask path.
4. The progress report marks the derived-mask train step as true.
5. The full Nix e2e gate passes.

## First Slice

Reuse the fixed `2x784 -> 8 -> 10` classifier shape.

## Status

Completed. `mnist-train-step-derived-mask` accepts images, labels, and the
parameter tree with no `relu_mask` input. It derives the ReLU activation and f32
mask internally with `stablehlo.compare` and `stablehlo.select`, then returns
the same four updated parameter tensors plus loss as the explicit-mask fixture.
The progress report now marks this milestone true.
