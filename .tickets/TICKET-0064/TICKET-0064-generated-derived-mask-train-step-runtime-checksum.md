# TICKET-0064: Generated Derived-Mask Train-Step Runtime Checksum

## Problem

The tiny derived-mask train-step runtime fixture proves the arithmetic shape, but
it is still not generated classifier-shaped train-step execution. The report
therefore correctly keeps `direct_mnist_external_runtime` false.

## Goal

Generate an external runtime checksum for `mnist-train-step-derived-mask`
semantics and compare loss plus updated-parameter evidence against an oracle.

## In Scope

- Generate runtime lowering for the derived-mask train-step semantics.
- Include forward pass, softmax-style loss, derived ReLU mask, gradients, and
  one SGD update.
- Return checksum evidence that covers loss and updated parameters.
- Compare against deterministic oracle values.
- Decide explicitly whether the evidence is strong enough to flip
  `direct_mnist_external_runtime`.

## E2E Focus

Add a default-gate runtime case for generated derived-mask train-step execution.

## Acceptance Criteria

1. Generated train-step semantics execute through the external runtime path.
2. The returned checksum covers both loss and parameter updates.
3. Oracle comparison is deterministic and tolerance is explicit.
4. The readiness report can justify whether direct MNIST runtime remains false
   or flips true.
5. The full Nix e2e gate passes.

## First Slice

Start from the generated forward checksum and add the minimum backward/update
surface needed to mirror `mnist-train-step-derived-mask`.

## Status

Ready for analysis.
