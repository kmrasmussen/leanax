# TICKET-0045: Compare And Select Primitives

## Problem

The classifier train-step path still accepts an explicit `relu_mask` because
LeanAX cannot yet express the comparison and selection pattern needed to derive
that mask from pre-activations.

## Goal

Add a minimal checked comparison/select primitive slice for fixed-shape tensor
programs.

## In Scope

- A greater-than comparison that returns a predicate tensor.
- A select operation that chooses between same-shaped f32 tensors using a
  same-shaped predicate tensor.
- StableHLO text rendering for `stablehlo.compare` and `stablehlo.select`.
- Structural verifier and numeric oracle support.
- A manifested numeric `compare-select` case.

## E2E Focus

Add `compare-select` as a numeric manifest case with golden text, MLIR parsing,
lowering manifest validation, and deterministic Python oracle coverage.

## Acceptance Criteria

1. LeanAX can lower a comparison result with predicate dtype.
2. LeanAX can lower a select over f32 tensors using that predicate tensor.
3. The structural verifier allows the new StableHLO operations.
4. The Python oracle evaluates the new operations.
5. The full Nix e2e gate passes.

## First Slice

Support rank-2 f32 tensors only, matching the ReLU hidden activation shape.

## Status

Completed. LeanAX now has checked `compareGt` and `select` bindings that lower
to `stablehlo.compare` and `stablehlo.select`. The manifested
`compare-select` case covers golden text, lowering manifest validation, MLIR
parsing, structural text verification, and deterministic numeric oracle
execution.
