# TICKET-0035: Classifier Progress Report

## Problem

As the MNIST path grows, it will be easy to confuse fixture smoke training,
checked compiler artifacts, external runtime execution, and real classifier
training.

## Goal

Add a machine-checked or e2e-produced progress report that states exactly which
classifier milestones are currently satisfied.

## In Scope

- A small script or runner mode that inspects generated artifacts and manifest
  outcomes.
- Explicit booleans for forward, loss, gradient, train-step, fixture training,
  full-dataset loading, and external runtime coverage.
- Documentation that maps the report back to the roadmap phases.
- A stable e2e assertion so the report cannot silently drift.

## E2E Focus

Add a manifest case that runs the progress report and checks that the reported
milestones match the current ticket suite.

## Acceptance Criteria

1. One command prints the current MNIST classifier readiness state.
2. The report distinguishes fixture smoke, real ten-class classifier training,
   and direct external runtime execution.
3. The report is covered by the e2e runner.
4. The roadmap links to the report.
5. The full Nix e2e gate passes.

## First Slice

Emit a simple text or JSON report from the e2e Python layer, then tighten it as
the classifier artifacts land.
