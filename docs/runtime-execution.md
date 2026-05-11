# Runtime Execution

`TICKET-0018` is currently blocked on a practical external runtime.

The checked options so far are:

- Nix package attributes matching IREE: none found in the current Nixpkgs input.
- Nix package attributes matching StableHLO or XLA runtime tooling: none found.
- Dev-shell commands: no `iree-compile`, `iree-run-module`, `iree-run-mlir`,
  `stablehlo-opt`, `stablehlo-translate`, `mlir-cpu-runner`, or `lli`.
- LLVM MLIR tools: `mlir-runner` is present, but it runs lowerable CPU MLIR
  dialects and does not execute the current `stablehlo.*` generic-op modules.
- PyPI IREE packages: `uv` can install `iree-compiler` and `iree-runtime`, and
  Python imports work when `libstdc++` is visible, but the bundled
  `iree-compile` and `iree-run-module` executables are generic Linux binaries
  that fail under this NixOS environment with the dynamic-linker stub error.

The useful next route is probably one of:

1. package IREE or StableHLO in Nix for this project,
2. run PyPI IREE inside a reproducible FHS or `nix-ld` environment,
3. add a second lowering target for executable CPU MLIR and run it with
   `mlir-runner`.

Until one of those routes is implemented, LeanAX keeps the numeric oracle as the
value-checking gate and does not claim external runtime execution.

