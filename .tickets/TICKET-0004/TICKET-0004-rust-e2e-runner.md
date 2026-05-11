# TICKET-0004: Rust E2E Runner

## Problem

Manual commands are too easy to run inconsistently once Lean emits artifacts.

## Goal

Add a small Rust runner that finds the repository root, runs the Lean emitter,
and compares generated output to golden fixtures.

## In Scope

- `e2e/runner/Cargo.toml`
- `e2e/runner/src/main.rs`
- `e2e/manifest.txt`
- exact golden comparison

## Acceptance Criteria

1. `cargo run --locked --manifest-path e2e/runner/Cargo.toml` runs from the repo
   root.
2. `--repo PATH` can override root detection.
3. Generated output is compared exactly against `e2e/golden`.

## First Slice

Drive the `affine` case end to end.
