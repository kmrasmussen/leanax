# TICKET-0024: Parameter Tree And Optimizer Step

## Problem

The current training step updates one scalar-style parameter. MNIST MLP training
needs a structured set of weights and biases updated together.

## Goal

Represent a small parameter tree or record in the DSL and lower an SGD update
over all parameters.

## In Scope

- A simple parameter record for `w1`, `b1`, `w2`, and `b2`.
- DSL helpers for mapping updates over parameters.
- Golden and numeric checks for multi-parameter update behavior.
- Clear separation between pure LeanAX update logic and host-side parameter
  storage.

## E2E Focus

Add an `sgd-parameter-tree` numeric case that updates multiple parameters and
compares every output tensor against a Python oracle.

## Acceptance Criteria

1. LeanAX can name and validate a small structured parameter set.
2. A checked SGD update can produce updated weights and biases together.
3. `sgd-parameter-tree` is covered by golden text, MLIR parsing, and numeric
   oracle checks.
4. The host training-loop script can call the multi-parameter update fixture.
5. The full Nix e2e gate passes.

## First Slice

Start with two parameters, weight and bias, then expand to the two-layer MLP
record.

