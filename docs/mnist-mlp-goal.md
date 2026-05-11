# High-Level Goal: Train An MLP On MNIST

The compelling north star for LeanAX is:

> Write, transform, lower, and run enough Lean-native tensor code to train a
> small multilayer perceptron on MNIST.

That goal is deliberately ambitious. It is also concrete enough to keep the
project honest: LeanAX should eventually handle real data, real numeric kernels,
autodiff, parameter updates, and a runtime boundary.

## What Success Means

The first complete success case should be a modest MLP:

1. Load MNIST images and labels from an external data source.
2. Define a two-layer dense network in LeanAX.
3. Express forward pass, loss, gradients, and SGD-style updates.
4. Lower executable kernels through StableHLO/MLIR or a comparable runtime path.
5. Train for at least one short epoch and report loss/accuracy.
6. Keep Lean responsible for typed IR, validation, transformations, and selected
   proof obligations, not for replacing the external compiler/runtime.

This does not require matching JAX performance in the early versions. It does
require a real end-to-end training loop where the important compiler and
transformation boundaries are exercised.

## Stay Close To JAX

LeanAX should be recognizable to someone who already understands pure JAX. The
surface syntax will be Lean, and the implementation will be staged through a
typed Lean IR, but the conceptual program shape should stay close to:

1. define pure functions for `forward`, `loss`, `accuracy`, and `update`,
2. transform those functions with `jit`, `vmap`, and `grad`,
3. keep parameters and batches explicit,
4. avoid hidden mutable model state,
5. run a host-side training loop that repeatedly calls compiled steps.

The point is not to copy JAX syntax. The point is to preserve the useful JAX
discipline: small pure tensor functions plus explicit transformations. LeanAX
then adds static shape/dtype evidence, checked lowering, and eventually proofs
around selected transformations.

## Milestone Ladder

1. **Checked Tensor IR**
   Static dtypes, ranks, shapes, modules, and structured validation errors.

2. **More Primitive Ops**
   Constants, broadcast, reshape, transpose, reduce-sum, exp/log where needed,
   matmul variants, and scalar/tensor literals.

3. **Executable Lowering**
   Replace StableHLO-like text with real StableHLO/MLIR parsing and at least one
   external execution path.

4. **Numeric E2E Tests**
   Compare small LeanAX programs against Python oracles for values, not just text
   structure.

5. **Transforms**
   Add `vmap` for batching, then reverse-mode autodiff over the checked IR.

6. **Training Runtime**
   Add enough host-side orchestration to move data, initialize parameters, run
   forward/backward/update steps, and collect metrics.

7. **MNIST MLP**
   Train a small dense classifier end to end, with checked artifacts at each
   compiler boundary.

## Why MNIST MLP Is The Right Target

MNIST is small enough that the project can avoid distributed runtime concerns,
but rich enough to force the hard parts:

- data ingestion,
- batching,
- matrix multiplication,
- nonlinearities,
- loss functions,
- autodiff,
- optimizer updates,
- execution through a backend,
- and meaningful numerical verification.

Reaching this point would prove LeanAX is more than a pretty-printer. It would
show that Lean can serve as a serious typed front end for array programming while
delegating backend code generation to the ecosystem that already exists.

See [mnist-mlp-roadmap.md](mnist-mlp-roadmap.md) for the staged path from the
current prototype to that goal.
