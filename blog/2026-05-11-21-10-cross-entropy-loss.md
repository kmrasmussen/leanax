# Cross-Entropy Loss Slice

`TICKET-0021` is complete.

LeanAX now has a first classification-loss path: a fixed-shape two-class
softmax cross-entropy module. The implementation adds the primitive pieces
needed for this slice: `stablehlo.exponential`, `stablehlo.divide`, and
`stablehlo.log`, then composes them with reduce-sum, broadcast, multiply, and a
scalar `-1.0` constant.

The new `cross-entropy-loss` e2e case is intentionally small. It validates the
loss structure against deterministic logits and one-hot labels, not a full
batched MNIST loss yet. It runs through golden text comparison, MLIR parsing,
lowering manifest validation, and the Python numeric oracle.

The manifest also includes `bad-cross-entropy-shape`, which proves mismatched
label and logit dimensions are rejected before lowering.
