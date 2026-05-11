# LeanAX

LeanAX is a design exploration for a Lean-native array programming system inspired
by JAX, with compilation to XLA through a stable tensor IR such as StableHLO.

The working idea is not "JAX reimplemented in Lean" in a narrow sense. It is:

1. Use Lean as the host language for a small, typed tensor DSL.
2. Represent tensor programs in an explicit intermediate representation.
3. Support JAX-like transforms over that IR: `jit`, `grad`, `vmap`, and shape
   checking.
4. Lower the checked IR to StableHLO/MLIR, then rely on XLA for device code.
5. Use Lean's type system and theorem proving where they are actually useful:
   shape safety, transformation correctness, and semantic contracts.

## Why This Might Be Interesting

JAX gets much of its power from a clean split:

- Python is the user-facing language.
- Tracing captures array code into an IR.
- Transformations rewrite that IR.
- XLA compiles the transformed IR for CPU/GPU/TPU execution.

Lean could occupy the front-end and proof layer in that stack. Instead of tracing
ordinary Python at runtime, LeanAX could construct a typed expression graph
directly, with shape and dtype facts available before lowering.

## Non-Goals For The First Pass

- Do not build a full NumPy clone.
- Do not write a new device compiler.
- Do not target raw XLA internals first if StableHLO/MLIR is sufficient.
- Do not make theorem proving mandatory for every user expression.

## Initial Artifacts

- [docs/jax-xla-lean.md](docs/jax-xla-lean.md): what JAX and XLA are, and where
  Lean fits.
- [docs/compiler-sketch.md](docs/compiler-sketch.md): a possible LeanAX compiler
  pipeline.
- [docs/mnist-mlp-goal.md](docs/mnist-mlp-goal.md): the high-level north star:
  train a small MLP on MNIST through LeanAX.
- [docs/mnist-mlp-roadmap.md](docs/mnist-mlp-roadmap.md): a staged path from
  the current prototype to a JAX-shaped MNIST training loop.
- [examples/first-sketch.lean](examples/first-sketch.lean): pseudocode for the
  kind of API this project is imagining.
- [notes/dialogue-001.md](notes/dialogue-001.md): first set of discussion
  questions and decisions.
- [amitious_plan](amitious_plan): ambitious roadmap documents and phase-one
  planning.
- [.tickets](.tickets): concrete ticket backlog for implementation slices.

## Current E2E Slice

The repository now contains a minimal Lean package and a reproducible e2e path:

```sh
nix develop --command bash -lc 'lake build && cargo test --locked --manifest-path e2e/runner/Cargo.toml && cargo run --locked --manifest-path e2e/runner/Cargo.toml'
```

That command builds the Lean executable, tests the Rust manifest parser, emits
MLIR-parseable StableHLO-shaped text for the manifested examples, compares the
output against `e2e/golden`, runs a no-dependency Python verifier through `uv`,
parses generated modules with `mlir-opt --allow-unregistered-dialect`, executes
numeric cases through a small Python evaluator for the supported generated-op
subset, and checks a deterministic host-side training loop. The unified e2e
manifest records numeric cases, validation failures, and training-loop checks,
so unexpected successes and unexpected failures are checked by the same runner.

## A Useful First Milestone

The smallest meaningful prototype would be:

1. Define a Lean tensor expression IR with static shapes and dtypes.
2. Add primitive ops: constants, add, multiply, matmul, reshape, reduce-sum.
3. Pretty-print that IR as a StableHLO-like textual module.
4. Separately validate the output against real StableHLO tooling.
5. Add one source-to-source transform, probably `vmap` before `grad`.

That would prove the core idea without pretending that the hard parts are solved.

## North Star

The larger target is to train a small multilayer perceptron on MNIST from a
Lean-native tensor program. The program should stay close to how a pure JAX MLP
training loop is structured: pure `forward`, `loss`, and `train_step` functions,
explicit parameters and batches, and transforms such as `jit`, `vmap`, and
`grad`. LeanAX will differ by making shapes, dtypes, IR construction, and
selected correctness arguments explicit. The project should grow toward that
target in checked slices rather than jumping straight to a large runtime.
