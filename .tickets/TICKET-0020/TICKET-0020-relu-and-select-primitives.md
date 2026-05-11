# TICKET-0020: ReLU And Select Primitives

## Problem

The MNIST MLP forward path needs a nonlinearity. The current DSL uses a square
activation as a temporary stand-in, which is not representative of the roadmap
target.

## Goal

Add enough comparison/select support to express `relu(x) = max(x, 0)` in the
LeanAX IR and DSL.

## In Scope

- Scalar or tensor constants for zero.
- Comparison or maximum/select operations needed for ReLU.
- Validation rules for dtype and shape compatibility.
- StableHLO-shaped lowering for the chosen primitive representation.
- Golden, MLIR parse, and numeric oracle coverage.

## E2E Focus

Add a passing `relu-forward` case and at least one validation-failure case for a
bad ReLU/select shape or dtype.

## Acceptance Criteria

1. LeanAX can construct a checked ReLU over a ranked f32 tensor.
2. The DSL can use ReLU in a dense-layer forward example.
3. `relu-forward` is covered by golden text, MLIR parsing, and numeric oracle
   checks.
4. Invalid ReLU/select construction is covered by the manifest.
5. The full Nix e2e gate passes.

## First Slice

Implement tensor ReLU for one ranked f32 shape, then generalize validation once
the e2e case is stable.

## Status

Completed. LeanAX now has a checked `stablehlo.maximum` primitive, DSL ReLU
activation built from zero constant, broadcast, and maximum, and a `relu-forward`
case that runs through golden text comparison, MLIR parsing, lowering manifest
validation, and numeric oracle execution. The manifest also covers
`bad-maximum-shape` as an expected validation failure.
