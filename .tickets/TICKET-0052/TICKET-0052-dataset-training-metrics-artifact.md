# TICKET-0052: Dataset Training Metrics Artifact

## Problem

Training checks print human-readable summaries, but the readiness report cannot
inspect stable structured metrics for cached real-dataset runs.

## Goal

Emit a stable JSON metrics artifact for cached real-dataset training.

## In Scope

- Metrics for mode, split, sample count, batch count, first/final loss, and
  first/final accuracy.
- Artifact paths for the generated LeanAX modules used by the run.
- A verifier that fails on missing fields or nonsensical values.

## E2E Focus

Add a manifest case that validates the metrics artifact after the cached
real-dataset sweep.

## Acceptance Criteria

1. Metrics are machine-readable and deterministic enough for e2e checks.
2. The verifier checks schema and basic value ranges.
3. The metrics mention the derived-mask train-step artifact.
4. The full Nix e2e gate passes.

## First Slice

Write metrics into `generated/mnist-real-dataset-metrics.json` and validate it
with a small Python checker.

## Status

Completed. The cached training sweep writes
`generated/mnist-real-dataset-metrics.json` with schema
`leanax.mnist_dataset_metrics.v1`, cached train split metadata, loss and
accuracy metrics, sample and batch counts, and the generated artifact list. The
manifest now runs `mnist-dataset-metrics` immediately after the cached sweep so
`e2e/python/verify_mnist_dataset_metrics.py` checks the JSON schema, value
ranges, loss/accuracy behavior, referenced artifact paths, and the derived-mask
train-step artifact.
