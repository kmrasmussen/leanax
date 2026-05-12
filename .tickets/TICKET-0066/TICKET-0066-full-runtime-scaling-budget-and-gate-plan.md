# TICKET-0066: Full Runtime Scaling Budget And Gate Plan

## Problem

The generated runtime wave proved the operation surface with scaled
representatives, but the project does not yet know the artifact size, build
time, runner time, or checksum tolerance profile for exact-shape scalarized
runtime generation.

## Goal

Measure and document whether the existing LLVM `mlir-runner` route can carry
exact-shape MNIST classifier runtime artifacts in the default gate.

## In Scope

- Estimate scalar operation counts for exact-shape forward, loss, gradient, and
  train-step runtime artifacts.
- Add a small script or report that records projected generated line counts,
  runtime case size, and expected default-gate cost.
- Define thresholds for what belongs in the default manifest versus an opt-in
  smoke.
- Document checksum tolerances for exact-shape runtime comparisons.
- Decide whether to proceed with scalarized LLVM or introduce a loop/buffer
  lowering slice first.

## E2E Focus

Add a data-loader or planning verifier that fails if the recorded scaling plan
goes stale relative to the current classifier contract.

## Acceptance Criteria

1. The exact-shape runtime scale is quantified.
2. The default-gate budget decision is explicit.
3. The next runtime implementation ticket has a concrete target shape and
   expected checksum strategy.
4. The full Nix e2e gate passes.

## First Slice

Start by deriving counts from the existing fixed classifier contract and current
manifest metadata rather than generating huge artifacts.

## Status

Ready for analysis.
