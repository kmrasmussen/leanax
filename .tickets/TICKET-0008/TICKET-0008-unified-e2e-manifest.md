# TICKET-0008: Unified E2E Manifest

## Problem

The e2e runner currently reads successful cases from one manifest and expected
validation failures from a second manifest. That split makes future compiler
work harder to review because one feature can touch multiple test declarations.

## Goal

Use one manifest with explicit expected outcomes for every e2e case.

## In Scope

- Add outcome tags such as `pass` and `validation-fail`.
- Keep golden comparison and Python structural verification for passing cases.
- Keep stderr matching for expected validation failures.
- Fail if a validation-failure case unexpectedly writes an output artifact.
- Print a summary that separates pass cases from expected validation failures.

## Acceptance Criteria

1. `e2e/manifest.txt` contains both successful and validation-failure cases.
2. `e2e/invalid_manifest.txt` is no longer needed.
3. The Rust runner rejects unknown outcomes and malformed manifest lines through
   automated unit tests.
4. The full Nix e2e command passes.

## E2E Gate

```sh
nix develop --command bash -lc 'lake build && cargo test --locked --manifest-path e2e/runner/Cargo.toml && cargo run --locked --manifest-path e2e/runner/Cargo.toml'
```
