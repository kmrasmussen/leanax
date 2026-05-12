# TICKET-0040: Optional Full-Dataset Metrics Smoke

## Problem

The project has fixture metrics and an IDX parser, but it does not yet have an
opt-in command that trains on cached MNIST data and reports short-run metrics.

## Goal

Add an optional full-dataset smoke path that runs only when MNIST IDX files are
available locally.

## In Scope

- A command or script flag that selects cached full-dataset mode.
- A bounded sample/epoch option so local runs are fast and deterministic enough
  for development.
- Metrics for samples, batches, epochs, first/final loss, and accuracy.
- A skip-with-diagnostic behavior when cache files are absent.
- Documentation that this is opt-in and not part of the default Nix gate.

## E2E Focus

Keep the manifest network-free by testing the full-dataset mode with tiny local
IDX files, not by downloading MNIST.

## Acceptance Criteria

1. Fixture mode remains the default.
2. Cached full-dataset mode can run when IDX files are present.
3. The command reports stable short-run metrics.
4. Missing cache files produce a clear non-default diagnostic.
5. The full Nix e2e gate passes without network access.

## First Slice

Run the full-dataset path against a tiny local cache and prove metric output is
stable.

## Status

Completed. `mnist_train_command.py` now supports explicit cached IDX mode with
split, cache directory, explicit path, epoch, and sample-limit options. Missing
caches print a skip diagnostic instead of affecting the default fixture gate.
`mnist-full-dataset-smoke` builds a tiny local IDX cache, runs cached-train mode,
and verifies stable metric output without network access.
