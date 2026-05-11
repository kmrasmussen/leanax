# Day-One E2E Slice

The first useful capability is intentionally small:

1. Lean owns a tiny tensor IR.
2. Lean emits a StableHLO-like textual module for a named example.
3. A Rust runner invokes Lean, compares golden output, and calls a `uv` Python
   verifier.
4. A Nix flake provides Lean, Rust, and `uv` so the same command can run from a
   clean environment.

## Why This Slice

It proves the project can cross all important boundaries:

- Lean package builds.
- A Lean executable is usable from automation.
- Rust can drive the end-to-end path.
- Python can provide fast semantic checks without becoming the trusted compiler.
- Golden fixtures make regressions visible.

## Acceptance Gate

```sh
nix develop --command bash -lc 'lake build && cargo run --locked --manifest-path e2e/runner/Cargo.toml'
```

The command should generate a StableHLO-like file and verify it against the
golden fixture.
