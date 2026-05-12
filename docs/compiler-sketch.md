# Compiler Sketch

This is a concrete shape for a LeanAX implementation.

## Layer 1: User DSL

The user writes Lean code against a tensor API:

```lean
def f (x : Tensor Float32 #[128, 784]) (w : Tensor Float32 #[784, 10]) :
    Tensor Float32 #[128, 10] :=
  matmul x w
```

The first design fork is whether this code computes real Lean values or builds
an expression graph. For XLA compilation, it needs to build an expression graph.

Possible approaches:

- Use an embedded DSL where tensor values are symbolic expressions.
- Use Lean elaborator support and notation to capture a nicer syntax.
- Use quotation/commands for explicitly staged tensor programs.

The embedded DSL is probably the best first prototype.

For model training, the DSL should stay close to pure JAX structure even though
the syntax is Lean. A LeanAX MLP should be organized around pure functions such
as `forward`, `loss`, and `trainStep`, with explicit parameter values and batch
inputs. Transformations such as `jit`, `vmap`, and `grad` should be visible at
the same conceptual boundaries a JAX user would expect.

## Layer 2: Typed Tensor IR

The core IR should be small and boring:

```lean
inductive DType
  | f32
  | i32
  | pred

structure Shape where
  dims : List Nat

inductive Expr : DType -> Shape -> Type
  | param : String -> Expr dtype shape
  | const : Scalar dtype -> Expr dtype Shape.scalar
  | add : Expr dtype shape -> Expr dtype shape -> Expr dtype shape
  | mul : Expr dtype shape -> Expr dtype shape -> Expr dtype shape
  | matmul :
      Expr .f32 { dims := [m, k] } ->
      Expr .f32 { dims := [k, n] } ->
      Expr .f32 { dims := [m, n] }
  | reduceSum :
      Expr dtype shape ->
      Axis shape ->
      Expr dtype (shape.eraseAxis axis)
```

This style makes invalid programs hard to represent. The tradeoff is that
dependent shape programming can become heavy, so the real implementation may use
a checked untyped core plus proof-producing smart constructors.

## Layer 3: Transformations

JAX-like transforms become IR-to-IR functions:

- `jit`: mark a closed expression for lowering and backend compilation.
- `vmap`: add a batch axis and rewrite primitive ops across that axis.
- `grad`: produce an adjoint program for scalar-output functions.
- `shapeCheck`: elaborate dynamic shape constraints into static evidence.

The Lean advantage is that these transforms can have contracts:

```lean
theorem vmap_preserves_pointwise_semantics : ...
theorem grad_matches_forward_derivative : ...
```

The first implementation does not need those theorems everywhere. It should make
room for them in the architecture.

## Layer 4: Lowering

Lowering should target StableHLO/MLIR before raw XLA.

Prototype path:

1. Pretty-print a small StableHLO-like textual module.
2. Keep a simple mapping table from LeanAX primitives to StableHLO ops.
3. Run external tooling later to parse/verify the emitted module.
4. Add real MLIR emission only after the IR design survives examples.

Example mapping:

| LeanAX op | StableHLO-ish op |
| --- | --- |
| `add` | `stablehlo.add` |
| `mul` | `stablehlo.multiply` |
| `matmul` | `stablehlo.dot_general` |
| `compareGt` | `stablehlo.compare` |
| `select` | `stablehlo.select` |
| `reshape` | `stablehlo.reshape` |
| `reduceSum` | `stablehlo.reduce` |

## Layer 5: Runtime

There are two runtime choices:

- Lean only emits modules, and an external runner handles compilation/execution.
- Lean calls into a native library that owns the XLA/IREE/StableHLO runtime path.

For an early project, external runner is simpler and keeps the trusted Lean side
small.

## Prototype Roadmap

1. Typed IR with constants, parameters, elementwise add/mul, and matmul.
2. Shape-aware smart constructors with readable errors.
3. StableHLO-like text emitter.
4. Golden tests for emitted modules.
5. External verification through StableHLO/MLIR tools.
6. Numeric execution checks against Python/JAX-style oracles.
7. A JAX-shaped DSL example for a two-layer MLP forward pass.
8. `vmap` transform for elementwise ops and matmul.
9. Reverse-mode `grad` for a tiny scalar-output subset.
10. A host-side training loop that can eventually run MNIST.

## Main Risks

- Lean dependent types may make everyday tensor programming too verbose.
- StableHLO/XLA integration may dominate the project if tackled too early.
- Autodiff correctness is subtle, especially with broadcasting and reductions.
- Shape polymorphism is useful but can explode the complexity of the first IR.

The safest path is a small explicit IR, a tiny lowering target, and one transform
at a time.
