# StableHLO Verifier Boundary

`TICKET-0017` is complete.

The important finding is that the current Nix shell has LLVM `mlir-opt`, but not
`stablehlo-opt` or `stablehlo-translate`. That means LeanAX can keep proving that
generated files are valid MLIR generic syntax, but it cannot yet claim StableHLO
dialect semantic verification.

The e2e runner now makes that boundary executable. It probes for
`stablehlo-opt` at startup. If the tool is present, every passing generated
module goes through it after the existing Python structural verifier and generic
MLIR parse. If it is missing, the gate prints an explicit diagnostic and keeps
running the existing checks.

This is a small change, but it matters for the roadmap. Future runtime or
packaging work can add the real verifier without redesigning the runner; the
hook is already in the generated-module path.
