# TICKET-0044: Classifier Readiness Report V2

## Problem

The current progress report covers the first classifier milestone set. The next
phase will add command, full-dataset, monolithic artifact, and runtime
milestones that must not drift from the roadmap.

## Goal

Extend the readiness report so it closes out the next classifier phase with
explicit true/false milestones.

## In Scope

- Booleans for monolithic train step, command wrapper, cache resolver,
  full-dataset smoke, runtime capability matrix, dense runtime, and MNIST
  forward runtime.
- Roadmap links from each boolean to its ticket or phase.
- A closeout blog note that states which milestones remain false and why.
- Stable e2e assertion of the expected report.

## E2E Focus

Update the existing `mnist-progress-report` manifest case so drift fails during
the normal e2e gate.

## Acceptance Criteria

1. The report covers every ticket in the next phase suite.
2. The report output remains stable and machine-readable.
3. The roadmap and report agree on true and false milestones.
4. The closeout note records remaining hard gaps.
5. The full Nix e2e gate passes.

## First Slice

Add the new fields as false first, then flip them only in the implementation
commits that prove each milestone.
