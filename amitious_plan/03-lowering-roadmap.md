# Lowering Roadmap

LeanAX should target StableHLO/MLIR rather than inventing a backend ABI.

## First Lowering Target

Emit stable, deterministic, StableHLO-like text:

- `module @name`
- `func.func @main`
- tensor types such as `tensor<2x3xf32>`
- primitive ops such as `stablehlo.add`, `stablehlo.multiply`, and later
  `stablehlo.dot_general`
- explicit `return`

This is not yet full MLIR correctness. It is a stepping stone that lets the
project build deterministic e2e artifacts immediately.

## Next Gates

1. Add a stricter Python verifier for textual structure.
2. Add real `stablehlo-opt` or MLIR parser checks when tooling is available.
3. Emit source locations or comments that connect Lean IR nodes to output lines.
4. Add a JSON manifest beside generated modules for downstream tooling.
