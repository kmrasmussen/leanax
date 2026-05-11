# TICKET-0012: Numeric Oracle Checks

## Goal

Run small generated kernels and compare their results against Python/NumPy or
JAX-style oracle values.

## E2E Focus

The manifest should distinguish text-only cases from cases with numeric inputs
and expected outputs.

## Status

Completed. The unified e2e runner now supports `numeric` manifest cases that
emit checked modules, compare golden StableHLO-shaped text, parse it with MLIR,
execute the supported generated op subset in Python, and compare results against
case-specific oracle values.
