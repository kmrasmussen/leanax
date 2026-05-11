# LeanAX Ambitious Plan

This directory is the planning surface for turning LeanAX from a sketch into a
working Lean-native tensor front end with end-to-end verification.

The spelling `amitious_plan` is intentional because it matches the requested
project path. Each document aims to keep a high-level roadmap tied to concrete
e2e gates, so implementation work can move quickly without losing the larger
shape of the project.

## Documents

- [00-vision.md](00-vision.md): the long-term shape of LeanAX.
- [01-day-one-e2e.md](01-day-one-e2e.md): the first complete vertical slice.
- [02-ir-roadmap.md](02-ir-roadmap.md): tensor IR, types, and validation.
- [03-lowering-roadmap.md](03-lowering-roadmap.md): StableHLO/MLIR emission.
- [04-transform-roadmap.md](04-transform-roadmap.md): `vmap`, `jit`, and `grad`.
- [05-runtime-roadmap.md](05-runtime-roadmap.md): Python/Rust/backend boundary.
- [06-proof-roadmap.md](06-proof-roadmap.md): where proofs should enter.
- [07-e2e-roadmap.md](07-e2e-roadmap.md): the verification ladder.
- [08-ticket-breakdown.md](08-ticket-breakdown.md): phase-one tickets.

## Operating Rule

Every phase should end in an observable artifact:

1. a Lean executable or library feature,
2. a checked fixture or generated module,
3. a Rust or Python e2e gate that fails loudly,
4. a ticket state update that records what is next.
