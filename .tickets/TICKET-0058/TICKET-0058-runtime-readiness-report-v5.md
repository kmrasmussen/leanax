# TICKET-0058: Runtime Readiness Report V5

## Problem

The readiness report keeps direct MNIST external runtime false, but it does not
yet distinguish runtime operation inventory, scalar math runtime coverage, and a
tiny train-step runtime fixture.

## Goal

Extend the readiness report after the runtime hardening tickets land.

## In Scope

- Add booleans for runtime operation inventory, scalar math runtime coverage,
  and tiny derived-mask train-step runtime coverage.
- Keep `direct_mnist_external_runtime` false until the full classifier
  train-step semantics run externally.
- Document the remaining gap from tiny runtime fixture to full MNIST runtime.

## E2E Focus

Update `mnist-progress-report` so runtime-readiness drift fails in the normal
e2e gate.

## Acceptance Criteria

1. The report distinguishes partial runtime coverage from full direct MNIST
   runtime.
2. Direct runtime remains false unless the full external train-step gate exists.
3. Roadmap and ticket statuses agree with the report.
4. The full Nix e2e gate passes.

## First Slice

Land only after `TICKET-0055` through `TICKET-0057` establish the runtime
evidence the report should track.

## Status

Completed. `mnist-progress-report` now tracks runtime operation inventory,
scalar softmax-loss runtime coverage, and the tiny derived-mask train-step
runtime fixture. The report still keeps `direct_mnist_external_runtime` false
because the full classifier-shaped train-step artifact has not yet run through
an external runtime.
