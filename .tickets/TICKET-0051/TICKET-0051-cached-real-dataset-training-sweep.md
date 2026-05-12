# TICKET-0051: Cached Real Dataset Training Sweep

## Problem

The default real-dataset check still trains on only a tiny cached sample. That
is useful as a smoke, but it does not prove a meaningful cached IDX training
path.

## Goal

Add a bounded cached real-dataset training sweep that uses actual IDX samples
through the cache resolver.

## In Scope

- A new e2e training script that resolves the cached train split.
- A deterministic bounded sample count large enough to exceed the tiny fixture
  route while keeping CI/runtime practical.
- Stable loss and accuracy output.
- Clear failure when the cached IDX data is missing or malformed.

## E2E Focus

Add a manifest `training-loop` case for the cached real-dataset sweep.

## Acceptance Criteria

1. The script uses resolver-loaded IDX data, not the embedded fixture batch.
2. The script reports sample count, batch count, first/final loss, and
   first/final accuracy.
3. Runtime remains bounded for the default Nix e2e gate.
4. The full Nix e2e gate passes.

## First Slice

Use a modest deterministic sample count and reuse the existing NumPy training
loop implementation.
