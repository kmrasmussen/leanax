# TICKET-0054: Direct MNIST Runtime Boundary Plan

## Problem

Direct full MNIST external-runtime execution remains false, but the next
runtime step is not written down as a concrete implementation boundary.

## Goal

Produce a precise runtime boundary plan for the full classifier path.

## In Scope

- Document whether the next runtime route should be LLVM lowering expansion,
  StableHLO tooling, IREE, or another external runner.
- List the minimum operations and tensor shapes needed by
  `mnist-train-step-derived-mask`.
- Record what would count as direct full MNIST runtime execution.
- Break the chosen route into follow-up tickets.

## E2E Focus

This is a planning/documentation ticket, but it should cite the current runtime
capability matrix and generated classifier artifacts.

## Acceptance Criteria

1. The plan names the next runtime route and rejects at least one tempting but
   premature route.
2. The plan maps required operations to existing generated artifacts.
3. Follow-up ticket names are ready for the next roadmap expansion.
4. The full Nix e2e gate still passes.

## First Slice

Use the current `runtime-capability-matrix` output and derived-mask classifier
artifacts as the evidence base.

## Status

Completed. `docs/mnist-runtime-boundary.md` chooses LLVM lowering expansion
through the existing `mlir-runner` path as the next route, while rejecting
unpackaged StableHLO/IREE/FHS routes for the default gate until tooling is
reproducible. The plan maps the `mnist-train-step-derived-mask` inputs,
outputs, required operations, and key tensor shapes, defines what would count as
direct full MNIST runtime execution, and breaks the next runtime slice into
`TICKET-0055` through `TICKET-0058`.
