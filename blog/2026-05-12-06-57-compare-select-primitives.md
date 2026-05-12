# Compare And Select Primitives

`TICKET-0045` adds the smallest branchless choice primitive surface needed for
derived ReLU masks.

LeanAX now has checked `compareGt` and `select` bindings. The new
`compare-select` fixture compares two `2x3` f32 tensors, produces a predicate
tensor, and selects between the original values and thresholds.

The case is in the normal e2e manifest, so golden StableHLO-shaped text,
lowering manifest validation, MLIR parsing, structural text checks, and the
Python numeric oracle all exercise the new path.
