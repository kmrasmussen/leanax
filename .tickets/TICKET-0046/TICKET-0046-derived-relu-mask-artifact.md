# TICKET-0046: Derived ReLU Mask Artifact

## Problem

The first-layer gradient path needs a ReLU mask, but that mask is currently a
host-provided input rather than a checked LeanAX artifact.

## Goal

Add a small artifact that derives ReLU activation and an f32 mask from hidden
pre-activations.

## In Scope

- Input `hidden_pre : tensor<2x8xf32>`.
- Compare `hidden_pre > 0`.
- Select activated values between `hidden_pre` and zero.
- Select mask values between one and zero.
- Return both `hidden` and `relu_mask`.

## E2E Focus

Add `relu-derived-mask` as a numeric manifest case with oracle coverage for
positive, zero, and negative values.

## Acceptance Criteria

1. The artifact returns activated hidden values.
2. The artifact returns an f32 mask compatible with `grad-relu-dense`.
3. The oracle checks activation and mask values.
4. Documentation explains why the mask is now derivable.
5. The full Nix e2e gate passes.

## First Slice

Use the existing `2x8` hidden shape from the MNIST fixture classifier.
