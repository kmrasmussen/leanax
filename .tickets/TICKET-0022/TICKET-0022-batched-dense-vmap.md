# TICKET-0022: Batched Dense VMap

## Problem

The first `vmap` slice only covers pointwise expressions. Dense MLP training
needs batching behavior around matmul, bias broadcast, and activation.

## Goal

Extend `vmap` to the dense-layer patterns needed for a batched MLP forward pass.

## In Scope

- Batched matmul or dense-layer rewrite rules.
- Bias broadcast handling under a leading batch dimension.
- Shape preservation checks for batched dense modules.
- Numeric oracle coverage against a manually batched Python implementation.

## E2E Focus

Add a `vmap-dense` numeric case that transforms a per-example dense layer into a
batched module and compares it against a manually batched oracle.

## Acceptance Criteria

1. `vmap` supports the dense-layer subset needed by MLP forward.
2. Shape validation rejects unsupported batched matmul or broadcast patterns.
3. `vmap-dense` is covered by golden text, MLIR parsing, and numeric oracle
   checks.
4. Existing pointwise `vmap` coverage continues to pass.
5. The full Nix e2e gate passes.

## First Slice

Support one dense shape, such as `4 -> 3` over a batch of `2`, then generalize
the shape rule.

