# TICKET-0059: Runtime LLVM Codegen Skeleton And ABI

## Problem

The current runtime fixtures prove the LLVM `mlir-runner` route, but new cases
still rely on handwritten scalar LLVM snippets. That does not scale to the
classifier-shaped path because there is no shared generated runtime ABI,
registry shape, or lowering skeleton.

## Goal

Define the fixed-shape scalarized runtime ABI and codegen skeleton that later
runtime tickets can reuse.

## In Scope

- Specify the runtime function ABI, argument convention, and checksum return
  convention.
- Add a generated or manifest-driven tiny arithmetic runtime case that uses the
  new skeleton.
- Keep generated artifact names deterministic.
- Record enough metadata for the Rust e2e runner to execute the case and compare
  its checksum.
- Document the boundary between generated runtime lowering and handwritten
  fallback fixtures.

## E2E Focus

Add one manifested runtime case that is produced through the new skeleton and
executed through the existing `mlir-runner` gate.

## Acceptance Criteria

1. A shared runtime codegen/ABI skeleton exists and is used by at least one
   generated runtime fixture.
2. The Rust e2e runner can discover, execute, and compare the generated runtime
   case.
3. The implementation does not flip `direct_mnist_external_runtime`.
4. The full Nix e2e gate passes.

## First Slice

Start with the smallest arithmetic checksum that proves the ABI and registry
shape. Avoid pulling classifier tensor ops into this ticket.

## Status

Completed. Runtime LLVM cases now share a small fixed-shape scalar ABI renderer:
`llvm.func @main() -> f32`, scalar `f32` SSA lines, and one checksum returned
through `llvm.return`. The CLI runtime registry is now data-driven through
`runtimeLLVMCases`, and `generated-arithmetic-runtime` proves the skeleton in
the default manifest with an external `mlir-runner` checksum of `2.0`.
