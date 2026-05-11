# TICKET-0010: Neural-Network Primitive Ops

## Goal

Add enough primitive operations to express small MLP pieces: constants,
broadcast, reshape, transpose, and reduce-sum.

## E2E Focus

Each primitive needs at least one passing golden module and at least one relevant
validation-failure case when shape or dtype rules can be violated.

## Acceptance Criteria

1. The IR has bindings for constants, broadcast, reshape, transpose, and
   reduce-sum.
2. Checked constructors exist for each new primitive.
3. `nn-primitives` is a passing golden e2e case using every new primitive.
4. The manifest has validation-failure cases for invalid broadcast, reshape,
   transpose, and reduce-sum.
5. The full Nix e2e gate passes.

## E2E Gate

```sh
nix develop --command bash -lc 'lake build && cargo test --locked --manifest-path e2e/runner/Cargo.toml && cargo run --locked --manifest-path e2e/runner/Cargo.toml'
```
