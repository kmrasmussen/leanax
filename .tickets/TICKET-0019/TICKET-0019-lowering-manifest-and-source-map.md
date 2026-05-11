# TICKET-0019: Lowering Manifest And Source Map

## Problem

Generated MLIR text is deterministic, but downstream debugging has no structured
way to connect output operations back to LeanAX IR bindings.

## Goal

Emit a small machine-readable manifest beside generated modules that maps LeanAX
bindings, operation kinds, value names, and output locations.

## In Scope

- A JSON manifest for each generated passing module.
- Stable operation identifiers from LeanAX IR bindings.
- Runner checks that the manifest exists and references the generated module.
- Documentation for how downstream tools should consume the manifest.

## E2E Focus

The e2e runner should validate the manifest shape for every generated passing
case and compare at least one checked-in golden manifest.

## Acceptance Criteria

1. `emit-stablehlo` can write a sidecar manifest for generated modules.
2. The manifest records module name, input names, output names, and operation
   identifiers.
3. The Rust runner validates manifest shape for every passing generated case.
4. At least one sidecar manifest has a golden fixture.
5. The full Nix e2e gate passes.

## First Slice

Add sidecar output for `affine` and validate it before expanding to every case.

## Status

Completed. `emit-stablehlo` accepts `--manifest-out` and writes a JSON sidecar
with module, function, generated path, inputs, outputs, stable operation IDs,
operands, result types, and MLIR line numbers. The e2e runner validates a
sidecar for every passing generated module and compares the `affine` sidecar to
a checked-in golden fixture.
