# Runtime Codegen Skeleton

`TICKET-0059` starts the generated runtime wave with a small ABI and codegen
skeleton in `LeanAX/RuntimeLLVM.lean`.

Runtime cases can now use a shared renderer for `llvm.func @main() -> f32`,
scalar `f32` SSA instructions, and a checksum returned through `llvm.return`.
The runtime case registry is a data list, so later generated shape, reduce, and
dense cases can join the same CLI and e2e path without adding another
handwritten match arm.

The new `generated-arithmetic-runtime` case is intentionally tiny. It proves the
skeleton by emitting a helper-generated LLVM module, running it through
`mlir-runner`, and checking the checksum `2.0` in the default manifest. This
does not change the direct MNIST runtime claim.
