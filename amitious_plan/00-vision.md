# Vision

LeanAX should become a Lean-native array programming stack where the front end
knows enough about shapes, dtypes, and transformations to reject many bad tensor
programs before they reach an accelerator backend.

The project is not a new XLA. The intended split is:

```text
Lean DSL
  -> LeanAX tensor IR
  -> Lean transformations and validation
  -> StableHLO-like text, then real StableHLO/MLIR
  -> external compiler/runtime
```

## North Star

The ambitious version is a user writing a small Lean tensor program, applying
`jit`, `vmap`, or `grad`, and receiving a verified lowering artifact that can be
checked by external tooling. Proofs should first make the compiler harder to lie
about. They do not need to make every user program theorem-heavy.

The concrete product-shaped goal is to train a small MLP on MNIST from a
Lean-native tensor program. That target is large enough to require real compiler
and runtime boundaries: batching, matrix multiplication, nonlinearities, loss
calculation, reverse-mode autodiff, optimizer updates, data movement, and metric
reporting. It is also small enough to keep the first complete training story
within reach.

## Strategic Bets

- Keep the Lean core explicit and inspectable.
- Use Rust for durable e2e harnesses and filesystem/process orchestration.
- Use `uv` Python for quick oracle scripts, textual validators, and later
  comparisons against NumPy/JAX-style behavior.
- Push e2e slices early, even before the IR is beautiful.
- Introduce dependent types where they carry the most value, not everywhere at
  once.

## Milestone Shape

The project should advance in visible e2e gates:

1. deterministic text emission,
2. structural validation,
3. real StableHLO/MLIR validation,
4. numeric execution for small kernels,
5. batching and autodiff transforms,
6. optimizer and training-loop support,
7. MNIST MLP training.
