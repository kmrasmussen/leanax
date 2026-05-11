# IR Roadmap

The first LeanAX IR should be small enough to understand in one sitting and
structured enough to grow toward proofs.

## Phase 1

- DTypes: `f32`, `i32`, `pred`.
- Shapes: lists of natural dimensions.
- Tensor type: dtype plus shape.
- Values: named SSA-like references with tensor types.
- Bindings: parameter-free primitive operations over named values.
- Modules: named function, inputs, bindings, and returns.

## Phase 2

- Smart constructors that validate operand shapes.
- Structured error values instead of string-only failures.
- Literal constants and rank-polymorphic elementwise operations.
- Matmul and dot-general shape checks.

## Phase 3

- A typed expression graph with fewer invalid states.
- Proof-carrying validators for selected invariants.
- Clear separation between user DSL, checked IR, and lowerable IR.

## E2E Principle

Every new primitive should add at least one golden module and one harness check.
