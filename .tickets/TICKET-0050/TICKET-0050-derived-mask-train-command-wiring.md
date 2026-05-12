# TICKET-0050: Derived Mask Train Command Wiring

## Problem

`mnist_train_command.py` still reports and composes the explicit-mask
`mnist-train-step` artifact even though LeanAX now has
`mnist-train-step-derived-mask`.

## Goal

Make the derived-mask train-step the command-facing classifier update artifact.

## In Scope

- Emit or require `mnist-train-step-derived-mask` in the command artifact set.
- Keep explicit-mask `mnist-train-step` available as a compatibility fixture.
- Update command output so operators can see which train-step artifact is used.
- Update any artifact-composition checks that should prefer the derived-mask
  path.

## E2E Focus

The existing `mnist-train-command` manifest case should fail if the command
stops emitting or referencing the derived-mask train-step artifact.

## Acceptance Criteria

1. The train command uses the derived-mask train-step artifact by default.
2. The output artifact list includes `generated/mnist-train-step-derived-mask.mlir`.
3. The explicit-mask fixture remains manifested separately.
4. The full Nix e2e gate passes.

## First Slice

Change the command artifact registry and update expected output checks without
changing training math.

## Status

Completed. `mnist_train_command.py` now requires and reports
`generated/mnist-train-step-derived-mask.mlir` as the command-facing train-step
artifact. The artifact-composition check also requires the derived-mask artifact,
while the explicit-mask `mnist-train-step` remains a separate manifested numeric
compatibility fixture.
