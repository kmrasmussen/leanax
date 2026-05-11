# TICKET-0017: StableHLO Semantic Verifier

## Problem

`mlir-opt --allow-unregistered-dialect` proves the generated text is parseable
MLIR, but it does not prove StableHLO dialect semantics.

## Goal

Add the strongest practical StableHLO semantic verifier available in the Nix
shell, or document why the current environment cannot provide one yet.

## In Scope

- Investigate `stablehlo-opt`, `mlir-opt` dialect registration, or another
  packaged verifier route.
- Add tool detection to the e2e runner or flake check.
- Keep the existing generic MLIR parser gate as a fallback.
- Document the exact semantic coverage the selected verifier provides.

## E2E Focus

Every passing generated module should go through the verifier when the tool is
available. If no semantic verifier is available, the e2e gate should include an
explicit diagnostic that keeps the limitation visible.

## Acceptance Criteria

1. The Nix shell exposes the selected verifier tool, or the ticket records a
   precise blocker.
2. The Rust e2e runner invokes the verifier for every generated passing module
   when available.
3. The README and roadmap explain the difference between MLIR parsing and
   StableHLO semantic verification.
4. The full Nix e2e gate passes.

## First Slice

Probe the Nix package set for a real StableHLO verifier and wire the best
available check into the existing generated-module loop.

## Status

Completed. The local Nix shell exposes `mlir-opt` but not `stablehlo-opt` or
`stablehlo-translate`, and the Nix package set does not expose a direct
`stablehlo` attribute. The e2e runner now probes for `stablehlo-opt`, runs it on
every passing generated module when present, and otherwise emits an explicit
semantic-verifier-unavailable diagnostic while keeping the generic MLIR parser
gate. The README, roadmap, and `docs/stablehlo-verification.md` record the
coverage boundary.
