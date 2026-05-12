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

Current status: implemented as MLIR parser verification for passing generated
modules.

## Level 5: Numeric Oracle

Small examples run through an external runtime or Python oracle and compare
numeric outputs.

Current status: implemented for small kernels and MNIST-shaped generated
artifacts through Python oracles.

## Level 6: Runtime Execution

Generated artifacts run through a real backend path. The first cases should stay
small: elementwise expressions, matmul, and reductions.

Current status: partially implemented through LLVM `mlir-runner` scalar
fixtures, including scalar math and a tiny derived-mask train-step checksum.
The missing step is generated classifier-shaped runtime lowering.

## Level 7: Training Loop

LeanAX drives a tiny trainable model with external data, forward pass, loss,
gradients, optimizer updates, and metrics. The north-star case is a small MLP on
MNIST.

Current status: implemented on the host side with cached IDX data, a
derived-mask train command, and structured metrics. Direct external runtime for
the classifier-shaped train-step remains open.

## Current Frontier

The project is no longer blocked on the early ladder. Levels 1 through 5 are
covered in the default gate. Level 6 is proven for scalar and tiny train-step
runtime fixtures but not yet for generated classifier-shaped train-step
execution. Level 7 is covered by host-side training, with the direct runtime
backend still intentionally false.

The next useful e2e work is the generated runtime wave:

1. add a shared runtime codegen skeleton,
2. cover the remaining fixed-shape tensor ops,
3. generate a forward runtime checksum,
4. generate a derived-mask train-step runtime checksum,
5. update the readiness report based on that evidence.
