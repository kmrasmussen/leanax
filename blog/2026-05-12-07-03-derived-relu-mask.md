# Derived ReLU Mask Artifact

`TICKET-0046` removes the first small host-side shortcut from the ReLU gradient
path.

The new `relu-derived-mask` artifact takes `hidden_pre : tensor<2x8xf32>`,
computes `hidden_pre > 0`, and uses `stablehlo.select` twice: once for ReLU
activations and once for an f32 mask compatible with `grad-relu-dense`.

The numeric oracle includes positive, zero, and negative values, so strict
greater-than behavior is covered explicitly.
