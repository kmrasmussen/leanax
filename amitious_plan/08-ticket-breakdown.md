# Phase-One Ticket Breakdown

Phase one is the first vertical slice from planning to checked artifact.

## Tickets

1. `TICKET-0001`: Create planning and ticket scaffolding.
2. `TICKET-0002`: Add a reproducible Lean/Rust/uv project shell.
3. `TICKET-0003`: Implement the first Lean IR and StableHLO-like emitter.
4. `TICKET-0004`: Add the Rust e2e runner and golden fixture comparison.
5. `TICKET-0005`: Add the `uv` Python verifier and wire it into e2e.
6. `TICKET-0006`: Add Lean IR validation and an expected-failure e2e case.
7. `TICKET-0007`: Analysis - document the MNIST MLP north star.
8. `TICKET-0008`: Unify the e2e manifest around explicit expected outcomes.
9. `TICKET-0009`: Add structured validation errors and smart constructors.
10. `TICKET-0010`: Add constants, reshape, transpose, broadcast, and reduce-sum.
11. `TICKET-0011`: Parse generated modules with real MLIR tooling.

## Completion Rule

A ticket can be marked completed only when the artifact exists and the relevant
e2e command has passed.

## Completed Initial Slice

The first six tickets are implemented together as a vertical slice:

- Lean package and CLI.
- StableHLO-like emitters for `affine` and `matmul`.
- Golden fixture comparison from Rust.
- `uv` Python structural verification.
- Lean validation that rejects `bad-add-shape` before lowering.
- A unified e2e manifest that keeps pass and validation-failure cases in one
  regression gate.
- Structured validation errors rendered through stable CLI messages.
- Neural-network primitive coverage for constants, broadcast, reshape,
  transpose, and reduce-sum.
- MLIR parser verification for passing generated modules via `mlir-opt`.

## Next Ticket Themes

The next phase should make the MNIST MLP target less distant:

1. Add structured Lean validation errors and smart constructors.
2. Add constants, reshape, transpose, broadcast, and reduce-sum.
3. Validate generated text with real StableHLO/MLIR tooling.
4. Add numeric oracle checks for small kernels.
5. Design a JAX-shaped LeanAX DSL surface for `forward`, `loss`, and
   `trainStep`.
6. Add a first `vmap` transform over elementwise modules.
7. Add reverse-mode autodiff for scalar losses.
8. Add a minimal host-side training loop.

`TICKET-0007` is intentionally an analysis ticket. It captures the north-star
direction and roadmap; future tickets should break implementation work out of
that analysis.

## Roadmap Ticket Queue

These tickets break the roadmap into implementation slices. Each ticket should
land with an e2e gate that covers the behavior it introduces.

1. `TICKET-0008`: Unified e2e manifest and outcome summary.
2. `TICKET-0009`: Structured validation errors and smart constructors.
3. `TICKET-0010`: Constants, reshape, transpose, broadcast, and reduce-sum.
4. `TICKET-0011`: Real StableHLO/MLIR parser verification.
5. `TICKET-0012`: Numeric oracle checks for small kernels.
6. `TICKET-0013`: JAX-shaped DSL surface for MLP forward/loss/train-step.
7. `TICKET-0014`: First `vmap` transform.
8. `TICKET-0015`: Reverse-mode `grad` for scalar losses.
9. `TICKET-0016`: Minimal host-side training loop.
