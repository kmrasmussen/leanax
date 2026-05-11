# E2E Roadmap

The e2e ladder should become progressively more external and less self-referential.

## Level 1: Golden Text

Lean emits deterministic text. Rust compares it to checked-in golden files.

## Level 2: Python Structural Verifier

`uv` runs a no-dependency Python validator that checks module shape, required
ops, balanced braces, and basic SSA references.

## Level 3: Lean Validation

Lean rejects invalid IR before lowering. The runner includes expected-failure
cases and checks error messages.

## Level 4: Real StableHLO Tooling

When available, generated MLIR is parsed by StableHLO/MLIR tools.

## Level 5: Numeric Oracle

Small examples run through an external runtime or Python oracle and compare
numeric outputs.

## Level 6: Runtime Execution

Generated artifacts run through a real backend path. The first cases should stay
small: elementwise expressions, matmul, and reductions.

## Level 7: Training Loop

LeanAX drives a tiny trainable model with external data, forward pass, loss,
gradients, optimizer updates, and metrics. The north-star case is a small MLP on
MNIST.

The current implemented state is Level 1 through Level 3. The next major step is
Level 4: real StableHLO/MLIR tooling.
