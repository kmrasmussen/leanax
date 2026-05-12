# TICKET-0043: MNIST Forward Runtime Slice

## Problem

The MNIST forward artifact is checked by text, MLIR parsing, manifests, and
Python numeric oracles, but it has not run through an external runtime path.

## Goal

Add the first external runtime slice for the MNIST-shaped forward computation
when the available runtime path can execute it deterministically.

## In Scope

- A fixed classifier-forward runtime fixture with deterministic parameters and
  input batch.
- Runtime output reduced to a stable checksum or small tensor result.
- Comparison against the existing Python forward oracle.
- Clear fallback documentation if direct StableHLO execution remains blocked.

## E2E Focus

Add a runtime manifest case only once the generated fixture can run externally
inside the Nix shell.

## Acceptance Criteria

1. The runtime fixture covers the `2x784 -> 8 -> 10` forward shape or a clearly
   documented smaller stepping-stone shape.
2. The external runtime result matches the oracle.
3. The progress report distinguishes MNIST-forward runtime from full train-loop
   runtime.
4. Runtime docs explain the selected execution path and remaining gaps.
5. The full Nix e2e gate passes.

## First Slice

Prototype a checksum-style forward runtime fixture and only manifest it after
the Nix shell can execute it reliably.

## Status

Completed. LeanAX now emits `mnist-forward-runtime`, a small dense-ReLU-dense
LLVM runtime checksum that mirrors the MNIST forward pattern as a stepping-stone
shape. The manifest runs it through `mlir-runner` and compares the scalar result
against `66.6125`. This is documented as MNIST-forward runtime coverage, not
full `2x784 -> 8 -> 10` runtime execution.
