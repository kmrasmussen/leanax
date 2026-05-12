# TICKET-0055: Runtime Operation Inventory Verifier

## Problem

The runtime boundary plan lists the operations needed by
`mnist-train-step-derived-mask`, but drift in the generated artifact would not
currently fail the e2e gate.

## Goal

Add a verifier that inventories the derived-mask train-step artifact and checks
that the runtime operation surface remains explicit.

## In Scope

- Read `generated/mnist-train-step-derived-mask.mlir` or its lowering manifest.
- Check the required input and output tensor shapes.
- Check the operation set against the runtime boundary document.
- Print a stable operation and shape summary for follow-up runtime work.

## E2E Focus

Add a manifest case that runs the inventory verifier after the derived-mask
artifact is generated.

## Acceptance Criteria

1. The verifier fails if the train-step operation set changes unexpectedly.
2. The verifier fails if the input/output contract drifts.
3. The verifier output names the unsupported operation surface for LLVM runtime
   expansion.
4. The full Nix e2e gate passes.

## First Slice

Use the lowering manifest when possible so the check does not depend on fragile
text parsing.

## Status

Completed. `e2e/python/runtime_operation_inventory.py` reads
`generated/mnist-train-step-derived-mask.mlir.manifest.json`, checks the
train-step input/output contract, verifies the expected thirteen-operation
surface, and prints stable operation counts plus the currently unsupported
runtime expansion surface. The manifest runs it as `runtime-operation-inventory`
after the derived-mask artifact has been generated.
