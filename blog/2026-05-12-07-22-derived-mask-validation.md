# Derived Mask Validation Suite

`TICKET-0048` adds the first negative checks for the new compare/select surface.

`bad-compare-shape` proves `stablehlo.compare` operands must have matching
tensor types. `bad-select-predicate-shape` proves a predicate tensor must match
the selected f32 tensor shape.

Both cases are manifest `validation-fail` entries, so the e2e runner checks the
stderr snippets and also verifies no invalid MLIR file is left behind.
