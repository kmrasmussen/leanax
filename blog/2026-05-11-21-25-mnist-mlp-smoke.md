# MNIST MLP Smoke

`TICKET-0026` is complete.

The new smoke command is:

```sh
nix develop --command uv run --no-managed-python --python python3 --project e2e/python python e2e/python/mnist_mlp_smoke.py
```

It is intentionally fixture-mode only. The script uses the deterministic
MNIST-shaped data path, checks that generated LeanAX loss and optimizer artifacts
exist, trains a tiny parity classifier, prints stable loss and accuracy metrics,
and fails if loss does not decrease or accuracy regresses.

This is not full MNIST runtime execution yet. The external runtime blocker from
`TICKET-0018` still applies. The value of this ticket is that the host-side
training command now has a checked smoke shape and is included in the e2e gate.
