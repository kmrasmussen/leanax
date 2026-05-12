# TICKET-0039: MNIST Cache Resolver And Split Loader

## Problem

LeanAX can parse IDX bytes, but local full-dataset runs still need a consistent
way to find train/test files and choose a split without putting downloads in
the default e2e path.

## Goal

Add a host-side resolver for cached MNIST IDX files and expose train/test split
loading through the same batch contract as the fixture.

## In Scope

- Explicit path environment variables or CLI flags for image/label IDX files.
- Default cache discovery at `$XDG_CACHE_HOME/leanax/mnist` or
  `~/.cache/leanax/mnist`.
- Train/test split names for the four canonical MNIST IDX files.
- Clear diagnostics for missing, partial, or malformed caches.
- A tiny local IDX fixture that exercises resolver behavior without network
  access.

## E2E Focus

Add a data-loader manifest case that builds a temporary cache with tiny IDX
files, resolves a split, and verifies the returned batch contract.

## Acceptance Criteria

1. The resolver can load train and test splits from explicit paths.
2. The resolver can load train and test splits from the documented cache layout.
3. Missing or partial caches fail with actionable messages.
4. The default e2e gate remains network-free.
5. The full Nix e2e gate passes.

## First Slice

Implement cache discovery and split selection against tiny local IDX files.
