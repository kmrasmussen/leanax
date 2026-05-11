# Dialogue 001: What Are We Really Building?

## My Current Understanding

You want to explore whether the ideas behind JAX can be transplanted into Lean:
array programming, transformations like `grad` and `vmap`, and compilation to
XLA-style backends.

The important conceptual move is that JAX is not "fast NumPy" only. It is a
staging and transformation system:

1. Write array code in a host language.
2. Capture that code as an intermediate representation.
3. Transform the representation.
4. Compile the result to optimized device code.

LeanAX asks: what if the host language were Lean, and the captured representation
were statically typed enough that many array mistakes are impossible or provably
ruled out?

## First Position

Lean should probably not try to be the low-level tensor compiler. XLA, IREE, and
MLIR already live there.

Lean should instead be:

- the language for writing staged tensor programs,
- the checker for shapes and dtypes,
- the transformation engine for a small trusted IR,
- the place where selected transformations can be specified and proved.

## Questions For You

1. Are you more interested in practical compilation, or in proving correctness
   of tensor transformations?
2. Should the first version feel close to JAX ergonomically, or should it lean
   into being a more explicit typed DSL?
3. Do you want this project to become a real Lean package soon, or stay as a
   design notebook until the architecture is sharper?
4. Is the motivating use case machine learning, numerical kernels, verified
   compiler research, or learning how these systems fit together?

## Proposed Next Conversation

The next useful discussion is to separate three things that are easy to blur:

- JAX the Python library and user model.
- jaxpr/StableHLO/XLA as compiler representations and backends.
- Lean as a typed/proof-capable front-end for tensor programs.

Once those are separate, the design decisions become much less mysterious.
