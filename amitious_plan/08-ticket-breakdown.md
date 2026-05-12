# Ticket Breakdown

Phase one was the first vertical slice from planning to checked artifact. The
file now also records the later waves so the roadmap remains connected to the
active backlog.

## Phase-One Tickets

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
12. `TICKET-0012`: Execute generated modules against numeric oracles.
13. `TICKET-0013`: Add a JAX-shaped DSL and MLP forward example.
14. `TICKET-0014`: Add a first pointwise `vmap` transform.
15. `TICKET-0015`: Add a restricted scalar-loss gradient example.
16. `TICKET-0016`: Add a minimal host-side training loop check.

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
- Numeric oracle execution for generated affine, matmul, neural primitive, DSL,
  transform, gradient, and train-step modules.
- A checked two-layer MLP forward example built through the first DSL layer.
- A checked pointwise `vmap` example.
- A restricted `sum(x * x)` gradient example.
- A deterministic host-side synthetic training loop whose loss decreases.

## Next Ticket Themes

The completed roadmap queue makes the MNIST MLP target less distant:

1. Structured Lean validation errors and smart constructors.
2. Constants, reshape, transpose, broadcast, and reduce-sum.
3. Generated text validation with real StableHLO/MLIR tooling.
4. Numeric oracle checks for small generated kernels.
5. A JAX-shaped LeanAX DSL surface for a two-layer `forward`.
6. A first `vmap` transform over elementwise modules.
7. A restricted reverse-mode-style gradient module for scalar losses.
8. A minimal host-side training loop.

`TICKET-0007` is intentionally an analysis ticket. It captures the north-star
direction and roadmap; future tickets should break implementation work out of
that analysis.

## Roadmap Ticket Queue

These tickets broke the roadmap into implementation slices. Each landed with an
e2e gate that covers the behavior it introduces.

1. `TICKET-0008`: Unified e2e manifest and outcome summary.
2. `TICKET-0009`: Structured validation errors and smart constructors.
3. `TICKET-0010`: Constants, reshape, transpose, broadcast, and reduce-sum.
4. `TICKET-0011`: Real StableHLO/MLIR parser verification.
5. `TICKET-0012`: Numeric oracle checks for small kernels.
6. `TICKET-0013`: JAX-shaped DSL surface for MLP forward/loss/train-step.
7. `TICKET-0014`: First `vmap` transform.
8. `TICKET-0015`: Reverse-mode `grad` for scalar losses.
9. `TICKET-0016`: Minimal host-side training loop.

## Phase-Two Ticket Queue

The next queue starts from the completed phase-one slice and aims at the MNIST
MLP north star. These tickets should land one by one, each with an observable
e2e gate.

1. `TICKET-0017`: Add the strongest practical StableHLO semantic verifier.
2. `TICKET-0018`: Execute generated modules through an external runtime path.
3. `TICKET-0019`: Emit lowering manifests and source maps for generated modules.
4. `TICKET-0020`: Add ReLU and select-style primitives.
5. `TICKET-0021`: Add softmax cross-entropy loss coverage.
6. `TICKET-0022`: Extend `vmap` to batched dense-layer patterns.
7. `TICKET-0023`: Generate gradients for a tiny dense-model loss.
8. `TICKET-0024`: Represent parameter trees and multi-parameter SGD updates.
9. `TICKET-0025`: Add an MNIST-shaped host-side data loader.
10. `TICKET-0026`: Add the MNIST MLP training smoke command.

The intended order is verifier/runtime first, then model expressivity, then data
and the final training command. If runtime execution blocks, the model tickets
can still proceed against the current MLIR plus numeric-oracle gate while the
blocker is documented.

## MNIST Classifier And Runtime Frontier

The completed later tickets moved the project from a synthetic training loop to
real cached IDX data, generated MNIST-shaped artifacts, derived-mask train-step
semantics, structured metrics, and the first LLVM runtime fixtures.

Notable completed milestones:

1. Generated classifier forward and train-step artifacts.
2. Cached real-dataset training and metrics smoke checks.
3. Runtime capability and operation inventory reporting.
4. Scalar math runtime fixtures through `mlir-runner`.
5. Tiny derived-mask train-step runtime checksum.
6. Runtime readiness report v5 with `direct_mnist_external_runtime` still false.

The remaining gap is generated runtime execution for the classifier-shaped
train-step path, not another host-side training proof.

## Runtime Wave Ticket Queue

These tickets start from `TICKET-0058` and should land as small, reviewable
runtime slices.

1. `TICKET-0059`: Runtime LLVM Codegen Skeleton And ABI.
2. `TICKET-0060`: Runtime Shape Ops Lowering Fixtures.
3. `TICKET-0061`: Runtime Reduce Lowering Fixtures.
4. `TICKET-0062`: Runtime Dot/Dense Lowering Fixture.
5. `TICKET-0063`: Generated MNIST Forward Runtime Checksum.
6. `TICKET-0064`: Generated Derived-Mask Train-Step Runtime Checksum.
7. `TICKET-0065`: Runtime Readiness Report V6.

The queue should keep `direct_mnist_external_runtime` false until generated
classifier-shaped train-step semantics execute externally and the report can
explain that evidence precisely.
