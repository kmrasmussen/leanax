# TICKET-0042: Dense Kernel Runtime Fixture

## Problem

The only runtime-backed fixture is an affine checksum. It proves the external
LLVM runtime path exists, but it does not exercise classifier-like dense-layer
math.

## Goal

Add a runtime fixture for a dense or ReLU dense kernel that moves runtime
coverage closer to the classifier path.

## In Scope

- A generated runtime MLIR module for a deterministic dense-layer checksum.
- External execution through the available runtime path.
- A scalar expected value suitable for stable comparison.
- Documentation that this is dense-kernel runtime coverage, not full classifier
  runtime execution.

## E2E Focus

Add a `runtime` manifest case that emits the dense runtime fixture and compares
its external result against the deterministic expected value.

## Acceptance Criteria

1. LeanAX emits a dense-kernel runtime module.
2. The external runtime executes it in the Nix dev shell.
3. The result is compared against a deterministic oracle value.
4. The runtime progress report distinguishes this from full MNIST runtime.
5. The full Nix e2e gate passes.

## First Slice

Use a fixed small dense checksum before attempting classifier-forward runtime.

## Status

Completed. LeanAX now emits `dense-runtime`, an LLVM-dialect MLIR fixture for a
fixed `1x4 @ 4x3 + 3` dense checksum. The manifest runs it through
`mlir-runner` and compares the scalar result against `103.375`, expanding
runtime coverage beyond the affine checksum without claiming full MNIST runtime
execution.
