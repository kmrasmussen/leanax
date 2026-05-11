# Proof Roadmap

Proofs should enter where they remove real ambiguity from the compiler.

## Early Targets

- Tensor type rendering preserves dtype and shape information.
- Elementwise constructors preserve operand shape.
- Module validation guarantees return references are defined.
- Simple `vmap` preserves the body operation count and prepends one axis.

## Later Targets

- Lowering only emits references that exist in the source module.
- Dot-general shape validation implies output shape correctness.
- Transform-specific semantic preservation for a tiny denotational model.
- Proof-carrying artifacts that record which Lean checks were run.

## Non-Goal

Do not block the first e2e slices on a complete tensor semantics. The proof plan
should strengthen a working pipeline rather than replace it.
