# Transform Roadmap

Transforms are the reason LeanAX is interesting. They should operate over the
explicit IR rather than over arbitrary Lean execution.

## `jit`

The first `jit` can simply mark a closed module as lowerable. The e2e behavior is
"emit this named checked module".

## `vmap`

Start with elementwise `add` and `multiply`:

- add a leading batch dimension to inputs and outputs,
- preserve operation order,
- reject rank-incompatible operands with structured errors.

Then extend to matmul with conventional batched matmul rules.

## `grad`

Delay reverse-mode autodiff until the IR has scalar outputs and enough primitive
metadata. The first real gate should be a small scalar expression whose gradient
is checked against a Python finite-difference oracle.

## Proof Direction

The first proof target is not full autodiff correctness. It is a preservation
lemma for the simplest transform, such as elementwise `vmap` preserving output
shape.
