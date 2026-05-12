# Runtime Boundary Plan

`TICKET-0054` closes the phase-17 queue by making the direct MNIST runtime
boundary explicit.

The next route is LLVM lowering expansion through the existing `mlir-runner`
gate. That choice follows the current capability matrix: `mlir-opt` and
`mlir-runner` are available in the Nix shell, while `stablehlo-opt`,
`iree-compile`, and `iree-run-module` are still absent.

The new plan maps the `mnist-train-step-derived-mask` contract, including
inputs, outputs, intermediate shapes, and required operations. It also defines
what would count as direct full MNIST runtime execution: an external runtime
artifact must execute the train-step semantics and compare loss plus updated
parameter checksums against the existing oracle in the default e2e gate.

The next roadmap slice is now concrete: runtime operation inventory, scalar
math runtime coverage, a tiny derived-mask train-step runtime fixture, and a
runtime readiness report update.
