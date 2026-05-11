# TICKET-0018: External Runtime Execution

## Problem

The current numeric oracle executes the supported generated-op subset in a local
Python interpreter. That is useful, but it is not yet proof that lowered modules
can run through an external compiler or runtime path.

## Goal

Choose one practical runtime path and execute at least one small LeanAX-generated
module outside the handwritten Python oracle.

## In Scope

- Evaluate IREE, StableHLO reference tooling, or another available runtime.
- Keep the first supported examples small: `affine`, `matmul`, and one
  reduction case.
- Add runtime fixture inputs and expected outputs.
- Preserve the existing Python oracle as an independent comparison.

## E2E Focus

Add a manifest outcome for runtime-backed execution and make the runner compare
runtime outputs against deterministic expected values.

## Acceptance Criteria

1. The selected runtime command is available through the Nix shell or documented
   as a precise blocker.
2. At least one generated module runs through the external runtime path.
3. Runtime outputs are compared against the Python oracle or checked fixture
   values.
4. The full Nix e2e gate passes.

## First Slice

Try the smallest elementwise `affine` module first, because it avoids matmul and
reduction runtime complications.

## Status

Completed. IREE, StableHLO reference execution, and XLA runtime tooling remain
unavailable in the current Nix shell, so this ticket uses the practical local
external runtime path: LeanAX emits an executable LLVM-dialect MLIR module for
the `affine-runtime` fixture, and the e2e runner executes it with
`mlir-runner --entry-point-result=f32`.

The fixture mirrors the `affine` numeric oracle inputs, computes
`sum((x + bias) * (x + bias))`, and compares the external runtime output against
the deterministic expected value `94.25`. The e2e manifest now has a dedicated
`runtime` outcome, so runtime-backed execution is owned by the same full Nix
gate as golden comparisons, MLIR parsing, Python numeric oracles, validation
failures, data-loader checks, and training-loop checks.
