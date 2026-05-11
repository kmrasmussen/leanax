# Phase-One Completion Audit

The phase-one ticket queue is complete through `TICKET-0016`.

This closeout records the evidence instead of relying on ticket status files
alone. The backlog contains sixteen numbered ticket directories, and every
`ticket-state.json` is marked `completed`. The current `main` branch is aligned
with `origin/main`, so the committed work has been pushed.

## Ticket Evidence

- `TICKET-0001` created the planning and ticket scaffold in `amitious_plan/`
  and `.tickets/`.
- `TICKET-0002` added the reproducible Nix, Lean, Rust, and `uv` shell boundary.
- `TICKET-0003` added the first Lean IR and StableHLO-shaped emitter.
- `TICKET-0004` added the Rust e2e runner and golden comparison.
- `TICKET-0005` added the `uv` Python structural verifier.
- `TICKET-0006` added Lean validation and expected validation failures.
- `TICKET-0007` documented the MNIST MLP north star and staged roadmap.
- `TICKET-0008` unified the e2e manifest around explicit expected outcomes.
- `TICKET-0009` introduced structured validation errors and checked builders.
- `TICKET-0010` added neural-network primitive operations.
- `TICKET-0011` added MLIR parser verification through `mlir-opt`.
- `TICKET-0012` added numeric oracle execution for generated modules.
- `TICKET-0013` added the first JAX-shaped DSL layer and MLP forward case.
- `TICKET-0014` added a first pointwise `vmap` transform.
- `TICKET-0015` added a restricted scalar-loss gradient case.
- `TICKET-0016` added a deterministic host-side training loop check.

## E2E Evidence

The current gate is:

```sh
nix develop --command bash -lc 'lake build && cargo test --locked --manifest-path e2e/runner/Cargo.toml && cargo run --locked --manifest-path e2e/runner/Cargo.toml'
```

That gate builds LeanAX, runs the Rust manifest parser tests, emits the
manifested modules, compares them to goldens, runs the Python structural
verifier, parses generated MLIR text with `mlir-opt`, executes numeric oracle
cases, checks expected validation failures, and asserts the synthetic training
loop loss decreases.

The latest run completed with `8` numeric cases, `11` expected validation
failures, `1` training-loop case, and `0` unexpected outcomes.

## Next Work

The next ticket queue should move from phase-one coverage toward the MNIST MLP
target: richer DSL syntax, more transform coverage, real runtime execution, and
stronger StableHLO semantic verification.
