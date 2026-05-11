# JAX, XLA, And Where Lean Fits

## What JAX Is

JAX is a Python array programming system built around transformations of numeric
programs. Its user experience looks like NumPy, but the important idea is that
JAX can capture a Python function's array computation and transform it.

The famous transforms are:

- `jit`: compile a function for faster execution.
- `grad`: derive a function that computes gradients.
- `vmap`: turn a scalar or per-example function into a batched function.
- `pmap` / sharding APIs: distribute work across devices.

The key trick is that JAX does not treat Python execution as the final program.
For supported array operations, it traces execution into a compact intermediate
representation often called jaxpr. Transformations operate on that representation,
then compilation lowers toward XLA.

## What XLA Is

XLA is a compiler for numerical tensor programs. It takes a graph/IR made from
array operations and lowers it into optimized code for hardware targets such as
CPU, GPU, and TPU.

XLA is not a source language in the way Python or Lean is. It is closer to the
backend compiler in this stack. A front-end gives it tensor operations; XLA does
fusion, layout choices, algebraic simplifications, memory planning, and device
code generation.

In a modern compiler stack, a front-end often targets StableHLO or MLIR dialects
instead of depending directly on unstable backend internals. StableHLO is useful
because it gives front-ends a more portable operation set for tensor computation.

## What Lean Would Add

Lean is not primarily attractive here because it can run tensor kernels faster.
The interesting part is that Lean can describe programs with much more static
structure than Python.

LeanAX could use Lean for:

- Shape-indexed tensor types, such as `Tensor Float32 #[m, n]`.
- Dtype-aware operations that reject invalid programs before lowering.
- An explicit IR that does not require Python-style runtime tracing.
- Proofs that transformations preserve meaning, for selected transforms.
- Proof-carrying or proof-audited shape rewrites.
- A small, trusted lowering path from typed IR to StableHLO text/bytecode.

In this framing, Lean is the front-end, verifier, and transformation language.
XLA remains the optimizing backend.

## The Big Difference From JAX

JAX starts from dynamic Python and recovers a staged numeric program through
tracing. LeanAX can start from a staged numeric program directly.

That changes the feel:

- JAX: "Run this Python function with tracers, capture what happens."
- LeanAX: "Build this typed tensor expression, then lower it."

The Lean version can be less magical, more explicit, and more statically checked.
The cost is that it may feel less like ordinary host-language programming unless
the DSL is designed carefully.

## The Important Boundary

A LeanAX prototype should not try to replace XLA. The boundary should probably be:

```text
Lean source
  -> elaborated Lean DSL expression
  -> LeanAX typed tensor IR
  -> transformed LeanAX IR
  -> StableHLO / MLIR module
  -> XLA / IREE / compatible backend
  -> executable device code
```

This gives Lean a coherent job: make the front half trustworthy and expressive,
then hand the numeric backend to tools that already know how to generate fast
machine code.

## Open Questions

- Should LeanAX be an embedded DSL, a quoted DSL, or a custom syntax layer?
- Are shapes Lean-level types, values, or both?
- Should the first backend be StableHLO text, MLIR bytecode, or a simpler custom
  verifier format?
- Is reverse-mode autodiff implemented as a Lean IR transform, or delegated to a
  downstream autodiff dialect/tool?
- How much proof should be required for normal use?
