# Runtime Roadmap

LeanAX should keep runtime integration outside the trusted Lean core until the
front end is stable.

## Rust Responsibilities

- Locate the repo root.
- Invoke `lake` and Lean executables.
- Manage generated fixture paths.
- Compare golden files.
- Call optional external validators.
- Eventually launch StableHLO/IREE/XLA tooling.

## Python Responsibilities

- Lightweight text and manifest validators through `uv`.
- Numeric oracles for tiny examples.
- Later, optional comparisons with NumPy or JAX if dependencies are introduced.

## Lean Responsibilities

- Define IR.
- Validate IR invariants.
- Transform IR.
- Emit deterministic artifacts.

Keeping these responsibilities separate makes the e2e loop fast while leaving a
clear path to stronger validation.
