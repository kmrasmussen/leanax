# Phase-Two Ticket Queue

The phase-one queue reached a checked loop: Lean construction, golden output,
MLIR parsing, numeric oracle execution, expected validation failures, and a
host-side synthetic training check.

The next tickets turn the MNIST roadmap into concrete slices:

- stronger StableHLO semantic verification,
- an external runtime execution path,
- source maps for lowered artifacts,
- ReLU and classification loss support,
- batched dense-layer `vmap`,
- dense-model gradients,
- multi-parameter optimizer updates,
- MNIST-shaped host data,
- and a final `mnist-mlp` training smoke command.

The queue is intentionally e2e-first. Each ticket should add or extend a
manifested check instead of only adding internal library code. The runtime path
is the largest uncertainty; if it blocks, the model-shape tickets can still move
forward against the current MLIR parser and numeric-oracle gate while the
runtime blocker stays explicit.
