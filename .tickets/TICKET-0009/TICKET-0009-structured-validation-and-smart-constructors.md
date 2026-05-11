# TICKET-0009: Structured Validation And Smart Constructors

## Problem

Validation failures are currently string-only and many modules are assembled by
hand. That is enough for the first slice, but it will not scale to neural-network
expressions.

## Goal

Introduce structured validation errors and checked constructors that make valid
IR easier to build and invalid IR easier to test.

## E2E Focus

Every new error variant should have an expected-failure manifest case. Every
new constructor used by examples should be covered by at least one passing
golden case.

## Acceptance Criteria

1. Validation failures are represented as structured Lean values.
2. CLI validation errors still render stable human-readable messages.
3. Passing example modules use checked constructors.
4. The e2e manifest covers every added validation error variant.
5. The full Nix e2e gate passes.

## E2E Gate

```sh
nix develop --command bash -lc 'lake build && cargo test --locked --manifest-path e2e/runner/Cargo.toml && cargo run --locked --manifest-path e2e/runner/Cargo.toml'
```
