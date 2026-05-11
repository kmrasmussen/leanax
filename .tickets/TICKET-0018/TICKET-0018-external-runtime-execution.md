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

## Current Blocker

Runtime execution is not complete yet. The current Nixpkgs input does not expose
IREE, StableHLO runtime tooling, or XLA runtime tooling. The dev shell has
`mlir-runner`, but that does not execute the current `stablehlo.*` generic-op
modules. PyPI IREE packages install through `uv`, but their bundled
`iree-compile` and `iree-run-module` binaries fail on this NixOS environment
with the dynamic-linker stub error.

See `docs/runtime-execution.md` for the exact probe results and likely next
routes.

