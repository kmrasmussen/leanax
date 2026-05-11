# TICKET-0007: MNIST MLP North Star

## Problem

LeanAX has a working first vertical slice, but the long-term goal needs to be
specific enough to guide technical tradeoffs.

## Goal

Document the high-level project target: training a small MLP on MNIST from a
Lean-native tensor program, while relying on external StableHLO/MLIR/runtime
tooling for backend execution.

## In Scope

- Define what success means for the MNIST MLP goal.
- Connect the target to IR, transforms, lowering, runtime, and e2e milestones.
- Update project docs so future tickets can be prioritized against this target.

## Acceptance Criteria

1. The README names the MNIST MLP north star.
2. A dedicated goal document explains success criteria and milestone steps.
3. The ambitious plan connects the current e2e ladder to training.

## First Slice

Write the goal down clearly before expanding the implementation backlog.
