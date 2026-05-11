# TICKET-0025: MNIST Data Loader

## Problem

The current host training loop uses deterministic synthetic data. The north-star
case needs real MNIST-shaped images and labels without putting data loading into
the trusted Lean core.

## Goal

Add a host-side MNIST data path that can produce normalized batches for the
LeanAX training runner.

## In Scope

- A small Python or Rust data-loading utility.
- Download, cache, or fixture strategy documented for reproducibility.
- Normalization and batching outside the Lean core.
- A tiny smoke fixture that does not require a full dataset download in every
  local test run.

## E2E Focus

Add a data-loader e2e check that proves batch shapes, dtype normalization, label
range, and deterministic fixture behavior.

## Acceptance Criteria

1. The data loader produces image tensors shaped like flattened MNIST batches.
2. Labels are represented in the format expected by the loss ticket.
3. The e2e gate has a fast fixture mode that avoids flaky network dependency.
4. Documentation explains how to run against the full dataset.
5. The full Nix e2e gate passes.

## First Slice

Use a tiny checked-in fixture or generated MNIST-shaped sample before adding an
optional full dataset path.

