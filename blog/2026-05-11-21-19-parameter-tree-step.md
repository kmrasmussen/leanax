# Parameter Tree Step

`TICKET-0024` is complete.

LeanAX modules can now return more than one tensor. The renderer keeps the old
single-result syntax for existing modules, but can emit multi-result function
types and return lines for parameter updates.

The new `sgd-parameter-tree` case updates a small weight matrix and bias vector
together:

```text
next_w = w - 0.1 * grad_w
next_b = b - 0.1 * grad_b
```

The e2e runner validates the lowering manifest, parses the generated MLIR, and
checks both returned tensors against a Python oracle. The manifest also includes
`bad-parameter-tree-shape`, so mismatched update shapes remain an expected
validation failure rather than slipping into lowering.
