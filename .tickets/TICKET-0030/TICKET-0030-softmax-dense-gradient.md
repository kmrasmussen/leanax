# TICKET-0030: Softmax Dense Gradient

## Problem

The current dense gradient covers a one-layer squared loss. MNIST classification
needs gradients from softmax cross entropy through the logits dense layer.

## Goal

Add a checked gradient artifact for the final dense layer of a classifier using
batched softmax cross entropy.

## In Scope

- Gradient of cross entropy with respect to logits.
- Gradients for final-layer weight and bias.
- Batch mean scaling.
- Numeric oracle comparison against an analytic Python implementation.
- Validation failures for mismatched logits, labels, and dense parameters.

## E2E Focus

Add `grad-softmax-dense` as a numeric manifest case and compare generated
gradients against an analytic oracle on deterministic fixture tensors.

## Acceptance Criteria

1. LeanAX emits a final-layer gradient artifact for batched classification.
2. The artifact returns gradients for `w2` and `b2`.
3. The oracle checks batch mean scaling, not just unnormalized sums.
4. At least one expected validation failure covers a shape mismatch.
5. The full Nix e2e gate passes.

## First Slice

Start with hidden activations supplied as an input tensor before chaining the
gradient through the first layer.

## Status

Completed. LeanAX now has a `grad-softmax-dense` artifact for the final
classifier layer. It takes hidden activations, logits, and one-hot labels;
computes `softmax(logits) - labels` with batch-mean scaling; returns `grad_w2`
and `grad_b2`; and is checked against a deterministic analytic Python oracle.
`bad-grad-softmax-dense-shape` covers mismatched logits/labels as an expected
validation failure.
