# TICKET-0011: Real StableHLO Verification

## Goal

Move beyond StableHLO-like text by running generated modules through real
StableHLO/MLIR parser tooling when available in the Nix shell.

Nixpkgs currently exposes MLIR tooling in this project shell but not a separate
StableHLO verifier package. This ticket therefore uses `mlir-opt
--allow-unregistered-dialect` as the first external parser gate. It verifies
that generated artifacts are valid MLIR generic syntax carrying StableHLO op
names; StableHLO dialect semantic verification remains a later hardening step.

## E2E Focus

The runner should keep fast structural checks but add an external parser gate
for generated modules.

## Acceptance Criteria

1. Generated modules use MLIR generic-op syntax for StableHLO-shaped ops.
2. The Nix shell includes `mlir-opt`.
3. The Rust e2e runner parses every passing generated module with
   `mlir-opt --allow-unregistered-dialect`.
4. Golden fixtures reflect the MLIR-parseable output.
5. The full Nix e2e gate passes.

## E2E Gate

```sh
nix develop --command bash -lc 'lake build && cargo test --locked --manifest-path e2e/runner/Cargo.toml && cargo run --locked --manifest-path e2e/runner/Cargo.toml'
```
