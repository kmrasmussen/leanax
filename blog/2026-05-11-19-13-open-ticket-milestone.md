# Open Ticket Milestone

The remaining phase-one implementation tickets are now covered by one e2e gate.

The new manifest shape distinguishes numeric cases from text-only pass cases.
For numeric cases, the Rust runner still emits checked LeanAX modules, compares
the output against checked-in goldens, runs the structural Python verifier, and
parses the generated text with `mlir-opt`. It then executes the supported
StableHLO-shaped op subset in `e2e/python/numeric_oracles.py` and compares the
result against deterministic oracle values.

The feature slices are intentionally small:

- `mlp-forward` proves the first JAX-shaped DSL layer can lower a two-layer MLP
  forward pass.
- `vmap-pointwise` proves the first pointwise batching transform.
- `grad-square-sum` proves the first restricted scalar-loss gradient program.
- `linear-train-step` proves a checked scalar update module.
- `synthetic-linear` proves a host-side training loop can make loss decrease on
  deterministic data.

This does not make LeanAX a full JAX or XLA replacement. It does move the
project from text-generation smoke tests to a broader checked loop: Lean
construction, golden output, MLIR parsing, numeric interpretation, expected
validation failures, and host-side training behavior are all exercised together.
