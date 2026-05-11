# ReLU Forward Slice

`TICKET-0020` is complete.

The new primitive is `stablehlo.maximum`. ReLU is built in the DSL as a checked
zero constant, a broadcast to the input shape, and a maximum against the dense
layer output. That keeps the first slice simple while still matching the MLP
roadmap better than the temporary square activation.

The new `relu-forward` e2e case covers:

- Lean construction through the dense-layer DSL,
- generated StableHLO-shaped MLIR golden text,
- lowering manifest validation,
- generic MLIR parsing,
- numeric oracle execution,
- and an expected validation failure for mismatched maximum operand shapes.

The full gate now has `9` numeric cases, `12` expected validation failures, and
the existing synthetic training-loop check.
