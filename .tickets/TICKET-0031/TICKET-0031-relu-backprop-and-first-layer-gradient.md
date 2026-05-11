# TICKET-0031: ReLU Backprop And First-Layer Gradient

## Problem

The classifier needs gradients through the hidden ReLU and first dense layer.
Current gradients stop before that part of the MLP.

## Goal

Add a checked first-layer gradient artifact for a small ReLU MLP classifier.

## In Scope

- ReLU derivative or mask semantics for the supported static-shape subset.
- Backpropagation from hidden activation gradients to pre-activation values.
- Gradients for first-layer weight and bias.
- Numeric oracle comparison against analytic Python values.
- Shape validation for hidden-gradient and ReLU-mask mismatches.

## E2E Focus

Add `grad-relu-dense` as a numeric manifest case and compare generated
gradients against a deterministic Python oracle.

## Acceptance Criteria

1. LeanAX can represent the ReLU gradient needed by the classifier.
2. The artifact returns gradients for `w1` and `b1`.
3. The numeric oracle covers positive, zero, and negative pre-activation cases.
4. At least one validation failure catches a hidden-layer shape mismatch.
5. The full Nix e2e gate passes.

## First Slice

Use the same small hidden dimension chosen for `mnist-forward`.
