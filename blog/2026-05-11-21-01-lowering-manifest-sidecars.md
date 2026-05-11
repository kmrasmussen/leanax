# Lowering Manifest Sidecars

`TICKET-0019` is complete.

Generated MLIR now has a small JSON sidecar when `emit-stablehlo` is called with
`--manifest-out`. The sidecar records the generated path, module name, function
name, input and output values, and an ordered operation list with stable IDs,
operation names, operands, result types, and MLIR line numbers.

The e2e runner writes and validates a sidecar for every passing generated case.
The `affine` sidecar is also compared against a checked-in golden fixture, which
gives the manifest format the same regression pressure as generated MLIR text.

This is not a full source map yet, but it is enough to connect downstream
runtime or verifier failures back to LeanAX IR bindings without scraping the
MLIR text by hand.
