# Runtime Execution Blocker

`TICKET-0018` is not complete yet.

The first pass looked for a practical external runtime path. Nixpkgs does not
currently expose an IREE, StableHLO, or XLA runtime package in this project
input. The dev shell has `mlir-runner`, but the generated LeanAX files currently
carry `stablehlo.*` generic ops and are not lowerable CPU MLIR that
`mlir-runner` can execute.

PyPI IREE was close but not enough. `uv` can install `iree-compiler` and
`iree-runtime`, and the Python modules import when `libstdc++` is visible. The
actual compiler and runner still invoke bundled generic Linux executables, which
fail under this NixOS environment with the dynamic-linker stub error.

The useful next move is to make the runtime environment real: package IREE or
StableHLO in Nix, run PyPI IREE inside a reproducible FHS or `nix-ld`
environment, or add an executable CPU-MLIR lowering path for `mlir-runner`.
Until then, LeanAX should keep the Python numeric oracle as its value-checking
gate and avoid claiming runtime execution.
