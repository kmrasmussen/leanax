# TICKET-0005: uv Python Structural Verifier

## Problem

Golden text comparison catches drift but does not explain whether the emitted
module still looks like the intended StableHLO subset.

## Goal

Add a no-dependency Python verifier run through `uv` and invoke it from the Rust
e2e runner.

## In Scope

- `e2e/python/pyproject.toml`
- `e2e/python/verify_stablehlo_text.py`
- Rust runner integration

## Acceptance Criteria

1. The verifier accepts the golden affine module.
2. The verifier rejects missing module/function/return structure.
3. The Rust runner calls the verifier for every generated case.

## First Slice

Check balanced braces, required module/function/return markers, and basic
StableHLO operation presence.
