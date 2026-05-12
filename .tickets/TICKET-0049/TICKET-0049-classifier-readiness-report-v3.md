# TICKET-0049: Classifier Readiness Report V3

## Problem

The readiness report needs to distinguish the compatibility explicit-mask train
step from the cleaner derived-mask train step.

## Goal

Extend `mnist-progress-report` with compare/select, derived ReLU mask, and
derived-mask train-step milestones.

## In Scope

- New booleans for compare/select support, derived ReLU mask artifact, and
  derived-mask MNIST train step.
- Roadmap notes that the explicit-mask train step remains a compatibility
  fixture.
- A closeout blog note listing the remaining false milestones.

## E2E Focus

Update the existing progress-report manifest case so report drift fails in the
normal e2e gate.

## Acceptance Criteria

1. The report covers all tickets in phase 16.
2. New true booleans are backed by manifest cases and generated artifacts.
3. Remaining false booleans are still intentional and documented.
4. The full Nix e2e gate passes.

## First Slice

Add the new fields only after the corresponding artifacts are manifested.

## Status

Completed. `mnist-progress-report` now covers the full phase-16 surface:
compare/select artifact support, derived ReLU mask artifact, derived-mask MNIST
train step, and compare/select validation failures. The report keeps full
real-dataset training and direct full MNIST external-runtime execution false.
