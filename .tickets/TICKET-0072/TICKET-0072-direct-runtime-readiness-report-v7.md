# TICKET-0072: Direct Runtime Readiness Report V7

## Problem

After the exact-shape runtime wave lands, the readiness report must say whether
LeanAX has truly reached direct MNIST external runtime or has a reproduced
scaling blocker.

## Goal

Update the readiness report, roadmap, docs, and ticket statuses after
`TICKET-0066` through `TICKET-0071`.

## In Scope

- Add fields for runtime scaling budget, tensor indexing helpers, exact-shape
  forward, exact-shape loss, exact-shape gradients, and exact-shape train-step.
- Flip `direct_mnist_external_runtime` only if `TICKET-0071` meets the documented
  definition.
- If the flag remains false, record the blocker precisely.
- Update roadmap docs so the next gap is clear.

## E2E Focus

Make readiness report drift fail in the default e2e gate.

## Acceptance Criteria

1. The report distinguishes exact-shape runtime milestones from representative
   milestones.
2. The direct runtime flag matches the evidence.
3. Roadmap and ticket statuses agree with the report.
4. The full Nix e2e gate passes.

## First Slice

Land only after exact-shape runtime evidence or a reproduced scaling blocker is
available.

## Status

Ready for analysis.
