# TICKET-0011: Real StableHLO Verification

## Goal

Move beyond StableHLO-like text by running generated modules through real
StableHLO/MLIR parser tooling when available in the Nix shell.

## E2E Focus

The runner should keep fast structural checks but add an external parser gate
for generated modules.

## Status

Ready for development.
