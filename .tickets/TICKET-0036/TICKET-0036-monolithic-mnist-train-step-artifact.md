# TICKET-0036: Monolithic MNIST Train-Step Artifact

## Problem

The current classifier train step is proven by composing several generated
artifacts in the Python e2e layer. That is useful evidence, but it does not yet
prove that LeanAX can lower the full classifier update as one inspectable
artifact.

## Goal

Add a fixed-shape `mnist-train-step` LeanAX artifact that performs one
classifier update for a `2x784` fixture batch and returns the updated parameter
tree. The first slice may accept an explicit ReLU mask, matching the current
first-layer gradient artifact.

## In Scope

- Inputs for `images`, `labels`, `relu_mask`, `w1`, `b1`, `w2`, and `b2`.
- Reuse of the existing hidden width `8` and class count `10`.
- Forward pass, ten-class cross entropy structure, final-layer gradient,
  first-layer ReLU gradient, and SGD update in one lowered module.
- Multiple returned tensors for `next_w1`, `next_b1`, `next_w2`, and `next_b2`.
- Numeric oracle comparison against the current artifact-composed Python train
  step.

## E2E Focus

Add `mnist-train-step` as a numeric manifest case with golden text, MLIR
parsing, lowering manifest validation, and deterministic oracle coverage.

## Acceptance Criteria

1. LeanAX lowers one monolithic MNIST train-step update artifact.
2. The artifact returns all four updated classifier parameter tensors.
3. The numeric oracle compares the returned parameters against the existing
   composed train-step implementation.
4. The progress report flips `monolithic_mnist_train_step` to `true`.
5. The full Nix e2e gate passes.

## First Slice

Keep the static `2x784 -> 8 -> 10` fixture shape and accept an explicit ReLU
mask. Deriving the mask inside the module should wait for a dedicated
comparison/select primitive slice.
