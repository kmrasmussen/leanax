# TICKET-0021: Softmax Cross-Entropy Loss

## Problem

MNIST training needs a classifier loss. The current scalar-loss examples are
useful for validating gradients, but they do not model classification.

## Goal

Add a small, explicit cross-entropy loss path for logits and labels.

## In Scope

- Primitive coverage needed for a stable softmax or log-softmax slice.
- Label representation for tiny classification fixtures.
- Shape validation for logits, labels, and scalar loss output.
- Numeric oracle coverage against a Python implementation.

## E2E Focus

Add a `cross-entropy-loss` numeric case with tiny logits and labels, plus
validation-failure cases for mismatched class dimensions.

## Acceptance Criteria

1. LeanAX can express a scalar classification loss from logits and labels.
2. Validation rejects mismatched logits and label shapes.
3. The generated loss module is checked by golden text, MLIR parsing, and a
   Python numeric oracle.
4. The roadmap documents any approximation or stability limitation.
5. The full Nix e2e gate passes.

## First Slice

Start with a tiny fixed-shape two-class loss fixture before generalizing class
counts.

## Status

Completed. `LeanAX/Loss.lean` adds a first two-class softmax cross-entropy
builder over f32 logits and one-hot labels. The lowering uses exponential,
reduce-sum, broadcast, divide, log, multiply, and scalar negation. The
`cross-entropy-loss` case is checked by golden text, MLIR parsing, lowering
manifest validation, and a Python numeric oracle. `bad-cross-entropy-shape`
covers mismatched label/logit dimensions as an expected validation failure.
