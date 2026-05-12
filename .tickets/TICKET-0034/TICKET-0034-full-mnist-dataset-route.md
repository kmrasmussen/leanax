# TICKET-0034: Full MNIST Dataset Route

## Problem

The current MNIST data path is intentionally fixture-only for reproducible e2e
runs. A real classifier needs an optional full-dataset route with caching and
documented behavior.

## Goal

Add an optional full MNIST dataset loader path for local classifier runs while
keeping fixture mode as the default e2e path.

## In Scope

- Download or user-supplied dataset strategy.
- Cache directory documentation.
- Train/test split handling.
- Normalization and one-hot labels matching the fixture contract.
- A non-network e2e check for loader parsing using a tiny local sample.

## E2E Focus

Add a data-loader case that validates the full-dataset code path against a tiny
local IDX-style sample or equivalent deterministic fixture, without requiring
network access.

## Acceptance Criteria

1. The loader can read or fetch full MNIST data through a documented command.
2. The default e2e gate remains network-free.
3. The fixture and full-dataset paths return the same batch contract.
4. Documentation explains cache location and failure modes.
5. The full Nix e2e gate passes.

## First Slice

Implement IDX parsing against a tiny checked-in sample before adding download
logic.

## Status

Completed. The host MNIST loader now parses canonical IDX image and label bytes
or files, validates the static LeanAX classifier shape contract, normalizes
images, and returns the same one-hot batched structure as the deterministic
fixture. The manifest includes `mnist-idx-sample`, a network-free data-loader
case built from tiny in-memory IDX bytes. `docs/mnist-data.md` documents cache
layout, opt-in full-dataset usage, and failure modes.
