# StableHLO Verification

LeanAX currently runs generated modules through three checks:

1. exact golden text comparison,
2. the repository Python structural verifier,
3. `mlir-opt --allow-unregistered-dialect`.

The third check is useful but narrow. It proves the file is valid MLIR generic
syntax carrying `stablehlo.*` operation names. It does not prove StableHLO
dialect semantics, because the current Nix shell exposes LLVM MLIR tooling but
does not expose `stablehlo-opt` or another StableHLO dialect verifier.

`TICKET-0017` makes that limitation visible in the e2e gate. The Rust runner now
probes for `stablehlo-opt`. When the tool is present, every passing generated
module is sent through it after MLIR parsing. When it is absent, the runner emits
an explicit diagnostic and continues with the generic MLIR parser plus the
existing structural and numeric checks.

The local probe result that motivated this ticket was:

```sh
nix develop --command bash -lc 'for tool in stablehlo-opt stablehlo-translate mlir-opt; do printf "%s=" "$tool"; command -v "$tool" || true; done'
```

Only `mlir-opt` was available. A later runtime or packaging ticket can strengthen
this by adding a real StableHLO package to the shell and letting the existing
runner hook exercise it automatically.

