# Runtime Roadmap

LeanAX should keep runtime integration outside the trusted Lean core until the
front end is stable.

## Current State

The project has moved past text-only and Python-only checks. The default e2e
gate now covers:

- deterministic generated classifier artifacts,
- MLIR parser verification for generated modules,
- Python numeric oracles for the MNIST-shaped path,
- cached IDX training and structured metrics,
- LLVM `mlir-runner` execution for scalar runtime fixtures,
- helper-generated runtime fixtures for broadcast, reshape, and transpose,
- helper-generated runtime fixtures for row-wise, all-elements, and
  keepdim-style reductions,
- a tiny derived-mask train-step runtime checksum.

The direct full MNIST runtime flag remains false. The current runtime fixtures
prove the external route and train-step arithmetic shape, but they are still
handwritten or tiny scalar expansions rather than generated execution of the
classifier-shaped train-step artifact.

## Rust Responsibilities

- Locate the repo root.
- Invoke `lake` and Lean executables.
- Manage generated fixture paths.
- Compare golden files.
- Call optional external validators.
- Launch local MLIR/LLVM runtime tools such as `mlir-runner`.
- Track runtime capability, oracle, and readiness report drift.

## Python Responsibilities

- Lightweight text and manifest validators through `uv`.
- Numeric oracles for tiny examples.
- Numeric oracles for generated classifier and train-step checksums.
- Dataset cache and metrics checks that keep host-side training honest.

## Lean Responsibilities

- Define IR.
- Validate IR invariants.
- Transform IR.
- Emit deterministic artifacts.
- Emit or describe enough lowering metadata for generated runtime cases.

## Runtime Expansion Surface

The named operation surface for `mnist-train-step-derived-mask` is now covered
by helper-generated runtime fixtures. The next gap is composition: generated
dot/dense lowering and then generated forward/train-step runtime checksums.

## Next Wave

`09-runtime-wave.md` defines the next queue. It moves from handwritten scalar
runtime fixtures to generated runtime checks in this order:

1. shared LLVM codegen skeleton and ABI,
2. shape-op fixtures (completed),
3. reduce fixtures (completed),
4. dot/dense fixture,
5. generated MNIST forward checksum,
6. generated derived-mask train-step checksum,
7. readiness report v6.

Keeping these responsibilities separate makes the e2e loop fast while leaving a
clear path to stronger validation.
