# TICKET-0041: Runtime Capability Matrix

## Problem

Runtime support is easy to overstate. The project currently has generic MLIR
parsing, optional StableHLO verifier probing, and one LLVM `mlir-runner`
fixture, but no direct classifier-shaped StableHLO runtime.

## Goal

Add a checked runtime capability matrix that records what the current Nix shell
can actually run.

## In Scope

- Probe availability for `mlir-opt`, `stablehlo-opt`, `mlir-runner`, and any
  packaged IREE or StableHLO execution command.
- Emit a stable text or JSON report.
- Document which capabilities are hard gates and which are optional probes.
- Keep unavailable direct StableHLO runtime support visible without failing the
  default gate.

## E2E Focus

Add a manifest-covered probe that asserts the known required tools are present
and records optional tools without pretending they are available.

## Acceptance Criteria

1. One command prints the runtime capability matrix.
2. Required tools for the current gate are asserted.
3. Optional direct StableHLO/IREE tools are reported explicitly when absent.
4. Runtime documentation links to the matrix.
5. The full Nix e2e gate passes.

## First Slice

Report `mlir-opt`, `mlir-runner`, and `stablehlo-opt` availability from inside
the Nix dev shell.

## Status

Completed. `runtime-capability-matrix` now runs under the manifest and prints a
stable JSON report for required `mlir-opt` and `mlir-runner` tools plus optional
`stablehlo-opt`, `iree-compile`, and `iree-run-module` probes. Missing optional
direct StableHLO/IREE tools remain visible without failing the default gate.
