# TICKET-0007: Analysis - MNIST MLP North Star

## Problem

LeanAX has a working first vertical slice, but the long-term goal needs to be
specific enough to guide technical tradeoffs.

This is an analysis ticket, not an implementation ticket. It records the
project direction that future development tickets should be derived from.

## Goal

Document the high-level project target: training a small MLP on MNIST from a
Lean-native tensor program, while staying close to the shape of a pure JAX MLP
training loop and relying on external StableHLO/MLIR/runtime tooling for backend
execution.

## In Scope

- Define what success means for the MNIST MLP goal.
- Connect the target to IR, transforms, lowering, runtime, and e2e milestones.
- Explain how LeanAX should stay conceptually close to JAX without copying its
  syntax.
- Update project docs so future tickets can be prioritized against this target.

## Acceptance Criteria

1. The README names the MNIST MLP north star.
2. A dedicated goal document explains success criteria and milestone steps.
3. A roadmap document lays out the path from the current prototype to MNIST MLP
   training.
4. The ambitious plan connects the current e2e ladder to training.

## First Slice

Write the goal and roadmap down clearly before expanding the implementation
backlog.
