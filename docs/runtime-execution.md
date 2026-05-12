# Runtime Execution

`TICKET-0018` now has a first practical external runtime path.

The checked options so far are:

- Nix package attributes matching IREE: none found in the current Nixpkgs input.
- Nix package attributes matching StableHLO or XLA runtime tooling: none found.
- Dev-shell commands: no `iree-compile`, `iree-run-module`, `iree-run-mlir`,
  `stablehlo-opt`, `stablehlo-translate`, `mlir-cpu-runner`, or `lli`.
- LLVM MLIR tools: `mlir-runner` is present. It does not execute the current
  `stablehlo.*` generic-op modules directly, but it does execute LLVM-dialect
  MLIR emitted by LeanAX for the first runtime fixture.
- PyPI IREE packages: `uv` can install `iree-compiler` and `iree-runtime`, and
  Python imports work when `libstdc++` is visible, but the bundled
  `iree-compile` and `iree-run-module` executables are generic Linux binaries
  that fail under this NixOS environment with the dynamic-linker stub error.

Future routes toward direct StableHLO runtime execution are still:

1. package IREE or StableHLO in Nix for this project,
2. run PyPI IREE inside a reproducible FHS or `nix-ld` environment,
3. lower the full supported StableHLO subset into executable CPU MLIR instead of
   only emitting the first affine checksum fixture.

The implemented first slice is intentionally narrow:

- Runtime LLVM cases now share a small ABI/codegen skeleton in
  `LeanAX/RuntimeLLVM.lean`: every generated runtime program emits
  `llvm.func @main() -> f32`, uses scalar `f32` SSA values, and returns one
  checksum through `llvm.return`. The CLI registry is a data list rather than an
  open-coded match so later generated runtime cases can join the same path.
- `generated-arithmetic-runtime` is the first helper-generated runtime case. It
  uses the shared skeleton, executes with `mlir-runner`, and returns the
  checksum `2.0`.
- `broadcast-shape-runtime`, `reshape-shape-runtime`, and
  `transpose-shape-runtime` are helper-generated fixed-shape indexing fixtures.
  They scalarize the target tensor order, apply weighted checksums so index
  drift is visible, and return `46.0`, `91.0`, and `86.0`.
- `reduce-row-runtime`, `reduce-all-runtime`, and `reduce-keepdim-runtime`
  scalarize row-wise reduction, all-elements reduction, and a keepdim-style
  rebroadcast checksum. The runner compares them against `36.0`, `21.0`, and
  `261.0`.
- `generated-dense-runtime` uses the shared runtime skeleton for a tiny
  generated `2x2 @ 2x2 + bias` dense composition and returns the weighted output
  checksum `15.25`. This case now exercises the reusable `RuntimeTensor`
  helper layer for row-major scalar refs, deterministic tensor constants, dense
  products, and weighted checksums while preserving the existing golden text.
- `generated-mnist-forward-runtime` uses the same skeleton for a scaled
  dense-ReLU-dense classifier-forward representative and returns the weighted
  logits checksum `-0.525`.
- `exact-mnist-forward-runtime` emits the full fixed
  `2x784 -> 8 -> 10` dense-ReLU-dense forward computation through the LLVM
  runtime path. The generated golden is 33,443 lines, returns the logits
  checksum `3.970676` under `mlir-runner`, and is cross-checked by
  `exact-mnist-forward-runtime-oracle`.
- `exact-mnist-loss-runtime` reuses the exact forward body, computes row-wise
  softmax cross-entropy for the fixed `2x10` labels, and returns mean loss
  `2.261078` under `mlir-runner`. `exact-mnist-loss-runtime-oracle` checks the
  manifest expectation against the deterministic Python oracle.
- `exact-mnist-gradient-runtime` derives the ReLU mask internally, computes
  exact-shape `grad_w2`, `grad_b2`, `grad_w1`, and `grad_b1`, and returns a
  checksum over loss plus all gradients. The 72,125-line golden returns
  `-131.4983` under `mlir-runner`; the oracle uses an explicit `5e-3` tolerance
  for the long f32 checksum accumulation.
- `generated-derived-mask-train-step-runtime` uses the skeleton for a scaled
  generated train-step representative with internal ReLU mask derivation,
  softmax loss, gradients, SGD updates, and a checksum over loss plus updated
  parameters. It returns `1.4609127`.
- `mnist-progress-report` now includes `runtime_readiness_v6`, which requires
  the codegen skeleton, shape-op fixtures, reduce fixtures, generated dense,
  scaled generated forward, exact-shape forward/loss/gradients, generated
  train-step, and a still-false direct MNIST runtime flag.
- `runtime-scaling-budget` records the exact-shape scalarized runtime budget for
  forward, loss, gradient, and train-step cases before the project commits to
  large checked-in artifacts.
- `lake exe leanax emit-runtime-llvm --case affine-runtime --out
  generated/affine-runtime.mlir` emits an executable LLVM-dialect MLIR module.
- The module hardcodes the same `affine` fixture values used by the Python
  numeric oracle and returns `sum((x + bias) * (x + bias))`.
- The e2e runner executes it with `mlir-runner --entry-point-result=f32` and
  compares stdout against `94.25`.
- `e2e/manifest.txt` records this as a `runtime` outcome, so the runtime check
  is part of the normal Nix e2e gate.
- `dense-runtime` adds a second LLVM runtime fixture: a fixed dense-layer
  checksum for one `1x4 @ 4x3 + 3` computation. The runner compares its scalar
  result against `103.375`.
- `mnist-forward-runtime` adds a small dense-ReLU-dense forward checksum. It is
  a documented stepping-stone shape, not the full `2x784 -> 8 -> 10` classifier
  artifact. The runner compares its scalar result against `66.6125`.
- `softmax-loss-runtime` adds a scalar softmax cross-entropy checksum using
  LLVM intrinsics for exp/log plus floating-point division. The runner compares
  its scalar result against `0.31326166`.
- `tiny-train-step-runtime` scalar-expands a tiny derived-mask train step with
  ReLU mask derivation, softmax loss, dense-layer gradients, SGD updates, and a
  checksum over loss plus updated parameters. The runner compares its scalar
  result against `7.939712`.

## Capability Matrix

`e2e/python/runtime_capability_matrix.py` records the runtime tools visible in
the current Nix shell. The default gate requires `mlir-opt` for MLIR parsing and
`mlir-runner` for the LLVM runtime fixture. It reports `stablehlo-opt`,
`iree-compile`, and `iree-run-module` as optional probes.

The manifest case `runtime-capability-matrix` runs this report during the normal
e2e gate. Missing optional direct StableHLO/IREE tools are reported explicitly
without pretending that classifier-shaped StableHLO runtime execution is
available.

This proves LeanAX can emit a module that runs through an external compiler/JIT
path available in the project shell. It is not yet direct StableHLO execution;
that remains a later hardening step when StableHLO/IREE/XLA runtime tooling is
packaged cleanly for this project.

See [mnist-runtime-boundary.md](mnist-runtime-boundary.md) for the direct MNIST
runtime boundary. The next runtime slice expands LLVM lowering through the
existing `mlir-runner` path before revisiting unpackaged StableHLO or IREE
routes.
