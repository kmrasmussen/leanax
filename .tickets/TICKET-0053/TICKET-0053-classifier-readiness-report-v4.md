# TICKET-0053: Classifier Readiness Report V4

## Problem

The readiness report still marks `full_dataset_training` false and does not
distinguish command-level derived-mask wiring from artifact-level availability.

## Goal

Extend the progress report for derived command wiring and cached real-dataset
training.

## In Scope

- A boolean for derived-mask train command wiring.
- A boolean for cached real-dataset training sweep coverage.
- A boolean for structured dataset-training metrics.
- Clear documentation of why direct full MNIST external runtime remains false.

## E2E Focus

Update the existing `mnist-progress-report` manifest case so report drift fails
in the normal e2e gate.

## Acceptance Criteria

1. The report covers the phase-17 tickets.
2. `full_dataset_training` is only flipped if the default gate proves the
   cached real-dataset sweep intended by this phase.
3. Direct full MNIST external runtime remains explicitly false.
4. The full Nix e2e gate passes.

## First Slice

Add new fields after TICKET-0050 through TICKET-0052 land.
