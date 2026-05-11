# TICKET-0028: MNIST-Shaped Forward Module

## Problem

The current two-layer MLP forward artifact is intentionally tiny
(`4 -> 3 -> 2`). It proves the DSL shape, but it does not exercise MNIST input
or ten-class logits.

## Goal

Add an MNIST-shaped forward module that lowers `batch x 784` images through a
small hidden layer and returns `batch x 10` logits.

## In Scope

- Inputs shaped `batch x 784`.
- Parameter tensors for `w1`, `b1`, `w2`, and `b2`.
- ReLU hidden activation.
- Output logits shaped `batch x 10`.
- Golden and numeric checks against a small deterministic fixture.

## E2E Focus

Add `mnist-forward` as a numeric manifest case with golden text, MLIR parsing,
lowering manifest validation, and Python oracle comparison.

## Acceptance Criteria

1. The DSL can express the MNIST classifier forward pass without raw IR assembly.
2. The generated module uses the expected dense, bias, ReLU, and logits
   operations.
3. The numeric oracle checks the generated logits for fixture inputs and
   parameters.
4. A validation-failure case catches at least one MNIST forward shape mismatch.
5. The full Nix e2e gate passes.

## First Slice

Keep the hidden dimension small, for example `8`, so generated goldens stay
reviewable while the public input and output shapes match MNIST.
