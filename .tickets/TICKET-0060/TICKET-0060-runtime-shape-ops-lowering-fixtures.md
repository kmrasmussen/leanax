# TICKET-0060: Runtime Shape Ops Lowering Fixtures

## Problem

The direct MNIST runtime boundary plan names broadcast, reshape, and transpose as
remaining classifier-shaped runtime surface. These operations are shape-moving
but still need fixed-shape runtime evidence before the full train-step can be
generated.

## Goal

Add generated LLVM runtime fixtures for broadcast, reshape, and transpose using
the shared runtime skeleton from `TICKET-0059`.

## In Scope

- Cover `stablehlo.broadcast_in_dim` with a checksum that proves expanded
  indexing.
- Cover `stablehlo.reshape` with a checksum that proves flattened storage order
  is preserved.
- Cover `stablehlo.transpose` with a checksum that proves index permutation.
- Add Python oracle values or equivalent deterministic expected checksums.
- Keep shapes tiny but representative of the MNIST classifier patterns.

## E2E Focus

Manifest the new runtime cases and run them through the same external
`mlir-runner` path as earlier runtime fixtures.

## Acceptance Criteria

1. Broadcast, reshape, and transpose each have generated runtime evidence.
2. Runtime checks compare against deterministic oracle values.
3. The readiness/reporting surface distinguishes these fixtures from full direct
   MNIST runtime.
4. The full Nix e2e gate passes.

## First Slice

Use tiny fixed-rank examples that mirror classifier indexing patterns without
trying to generate the whole forward pass.

## Status

Ready for analysis.
