# TICKET-0002: Reproducible Lean/Rust/uv Shell

## Problem

The plain shell may not have Lean, Lake, or a configured Rust toolchain. E2E
work needs a reproducible command boundary.

## Goal

Add a Nix flake and Lean package metadata so project checks can run with Lean,
Rust, Cargo, and `uv`.

## In Scope

- `flake.nix`
- `lean-toolchain`
- `lakefile.lean`
- `lake-manifest.json`
- README command documentation

## Acceptance Criteria

1. `nix develop --command lake build` builds the Lean package.
2. The flake dev shell includes `cargo`, `rustc`, and `uv`.
3. A flake check builds Lean and checks/runs the e2e harness.

## First Slice

Create the package skeleton and make the harness command reproducible through
Nix.
