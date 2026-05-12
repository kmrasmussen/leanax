# TICKET-0067: Runtime Tensor Indexing Codegen Helpers

## Problem

Current generated runtime cases are scalar representatives. Exact-shape runtime
generation needs reusable tensor indexing helpers so forward, loss, gradient,
and train-step artifacts are generated from structured shape logic instead of
handwritten scalar strings.

## Goal

Add runtime codegen helpers for fixed-shape tensor indexing, dense products,
broadcast-like reads, reductions, transposes, and weighted checksums.

## In Scope

- Represent fixed tensor shapes and row-major offsets in Lean runtime codegen.
- Generate deterministic SSA names for scalarized tensor elements.
- Generate reusable helpers for dense, elementwise, reduce, transpose, and
  checksum patterns.
- Keep existing runtime fixtures stable.
- Add at least one fixture that proves the helpers generate the same output as
  an existing generated representative.

## E2E Focus

Run the helper-generated fixture through the normal runtime manifest and compare
it against golden LLVM MLIR plus `mlir-runner` checksum.

## Acceptance Criteria

1. Tensor indexing helpers exist and are used by at least one runtime case.
2. The generated text is deterministic.
3. Existing runtime cases continue to pass.
4. The full Nix e2e gate passes.

## First Slice

Refactor the generated dense or generated forward representative onto the new
helpers before generating exact-shape artifacts.

## Status

Completed. `LeanAX/RuntimeLLVM.lean` now has a `RuntimeTensor` helper layer for
deterministic scalar names, row-major offsets, fixed-shape index enumeration,
tensor constants, dense products, reduction refs, transpose refs, ReLU
elementwise refs, and weighted checksums. `generated-dense-runtime` now uses the
helpers while continuing to match its golden LLVM MLIR and `mlir-runner`
checksum.
