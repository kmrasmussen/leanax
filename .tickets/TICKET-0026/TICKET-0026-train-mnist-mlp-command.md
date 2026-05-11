# TICKET-0026: Train MNIST MLP Command

## Problem

The roadmap north star needs one command that exercises the checked compiler
path, host data path, training loop, and metrics together.

## Goal

Add `leanax train mnist-mlp` or an equivalent runner command for a short MNIST
MLP training smoke run.

## In Scope

- CLI entrypoint for the MNIST MLP training story.
- Integration with the parameter tree, forward, loss, gradient, optimizer, and
  data-loader tickets.
- Metrics for loss and accuracy.
- Fast smoke-mode defaults suitable for e2e.
- Documentation of runtime/tooling limitations.

## E2E Focus

Add a training smoke case that runs for a tiny number of batches, prints stable
metrics, and asserts loss or accuracy moves in the expected direction.

## Acceptance Criteria

1. A documented command starts the MNIST MLP smoke training flow.
2. The flow exercises checked LeanAX lowering artifacts rather than bypassing
   the compiler path.
3. The e2e gate asserts deterministic smoke metrics.
4. README and roadmap show the command and its current limitations.
5. The full Nix e2e gate passes.

## First Slice

Wire the command to the small fixture mode from `TICKET-0025`, then expand to a
short full-dataset run once runtime behavior is stable.

## Status

Completed. `e2e/python/mnist_mlp_smoke.py` is the documented fixture-mode MNIST
MLP smoke command. It checks that generated LeanAX loss and optimizer artifacts
exist, trains a tiny deterministic parity classifier over the MNIST-shaped
fixture, prints stable loss/accuracy metrics, and fails if loss does not
decrease or accuracy regresses. The README and roadmap document the command and
the runtime limitation.
