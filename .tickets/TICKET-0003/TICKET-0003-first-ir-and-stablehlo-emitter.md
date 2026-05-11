# TICKET-0003: First IR And StableHLO Emitter

## Problem

LeanAX needs an executable artifact, not only API sketches.

## Goal

Implement a small Lean IR and a CLI that emits StableHLO-like text for named
examples.

## In Scope

- DTypes, shapes, tensor types, value references.
- Simple operation bindings.
- Deterministic renderer.
- `lake exe leanax emit-stablehlo --case affine --out generated/affine.mlir`.

## Acceptance Criteria

1. `lake build` compiles the Lean library and executable.
2. The `affine` case emits deterministic StableHLO-like text.
3. The generated text has a checked-in golden fixture.

## First Slice

Support an affine-style elementwise example using `stablehlo.add` and
`stablehlo.multiply`.
