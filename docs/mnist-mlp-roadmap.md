# Roadmap To MNIST MLP Training

This roadmap starts from the current LeanAX slice: a tiny Lean IR, validation,
StableHLO-like text emission, golden fixtures, and a Rust/Python e2e runner.

The goal is to reach a small MNIST MLP training loop while staying close to how
a pure JAX version would be structured.

## Reference Shape: Pure JAX

A pure JAX implementation would roughly look like this:

```python
def init(key):
    w1 = random.normal(key, (784, 128)) * 0.01
    b1 = jnp.zeros((128,))
    w2 = random.normal(key, (128, 10)) * 0.01
    b2 = jnp.zeros((10,))
    return (w1, b1, w2, b2)

def forward(params, x):
    w1, b1, w2, b2 = params
    h = jnp.maximum(x @ w1 + b1, 0.0)
    return h @ w2 + b2

def loss(params, x, y):
    logits = forward(params, x)
    return cross_entropy(logits, y).mean()

@jax.jit
def train_step(params, batch):
    x, y = batch
    grads = jax.grad(loss)(params, x, y)
    return tree_map(lambda p, g: p - lr * g, params, grads)
```

LeanAX should preserve this architecture:

- parameters are explicit values,
- batches are explicit inputs,
- model code is pure,
- `grad` derives the backward program,
- `jit` marks the train step for lowering,
- host code owns data loading and loop orchestration.

LeanAX will differ where Lean is useful: tensor shapes and dtypes should be
visible in the program, IR construction should be explicit enough to validate,
and lowering should produce artifacts that can be checked.

## Phase 0: Current State

Already implemented:

- Lean package and CLI.
- Minimal SSA-like tensor IR.
- `add`, `multiply`, and rank-2 `dot_general`.
- Constants, broadcast, reshape, rank-2 transpose, and scalar reduce-sum.
- Shape/type validation for current primitives.
- StableHLO-like text emission.
- MLIR generic-op syntax that parses with `mlir-opt
  --allow-unregistered-dialect`.
- Rust e2e runner with golden comparisons.
- Python structural verifier.
- Expected-failure validation case.
- Unified e2e manifest with explicit expected outcomes.
- Numeric oracle execution for the supported generated-op subset.
- A first DSL-built two-layer MLP forward module.
- A first pointwise `vmap` transform.
- A restricted scalar-loss gradient module for `sum(x * x)`.
- A minimal deterministic host-side training loop check.

Gate:

```sh
nix develop --command bash -lc 'lake build && cargo test --locked --manifest-path e2e/runner/Cargo.toml && cargo run --locked --manifest-path e2e/runner/Cargo.toml'
```

## Phase 1: Make The IR Usable

Goal: make small neural-network expressions representable without hand-assembling
every binding.

Work:

- Add structured validation errors instead of string-only errors.
- Add smart constructors for checked module construction.
- Add literals and constants.
- Add broadcast, reshape, transpose, and reduce-sum.
- Add stable names for scalar values and ranked tensors.
- Add golden examples for each primitive.

Exit gate:

- A LeanAX module can express `x @ w + b`, a reduction, and a scalar loss-like
  expression.

Current progress:

- The first primitive expansion is implemented as `nn-primitives`, with e2e
  coverage for constants, broadcast, reshape, transpose, and reduce-sum.

## Phase 2: Validate Real Lowering

Goal: stop relying only on StableHLO-like text.

Work:

- Add real StableHLO/MLIR parser verification in the e2e runner.
- Tighten emitted syntax until external tooling accepts it.
- Separate the internal IR from the lowerable StableHLO subset.
- Keep Python structural checks as fast smoke tests.

Exit gate:

- Generated `affine`, `matmul`, and reduction examples parse under real
  StableHLO/MLIR tooling.

Current progress:

- Generated modules now parse under `mlir-opt --allow-unregistered-dialect`.
  This is real MLIR syntax validation with StableHLO-shaped generic ops, not yet
  StableHLO dialect semantic verification.
- The e2e runner now probes for `stablehlo-opt` and will run it on every passing
  generated module when available. The current Nix shell does not provide that
  verifier, so the gate emits an explicit diagnostic and keeps the limitation
  visible.
- Generated modules now have validated lowering manifest sidecars. The manifests
  connect LeanAX binding order, operation names, operands, result names, result
  types, and MLIR line numbers for downstream debugging.

## Phase 3: Numeric Execution

Goal: prove generated programs compute expected values, not just valid text.

Work:

- Choose an execution route: IREE, StableHLO reference tooling, or another
  practical runtime path.
- Add small input fixtures.
- Compare outputs against Python/NumPy or JAX oracles.
- Start with elementwise, matmul, and reduce-sum.

Exit gate:

- LeanAX-generated kernels run externally and match a Python oracle on small
  arrays.

Current progress:

- The runner executes generated text for the supported op subset in
  `e2e/python/numeric_oracles.py` and checks affine, matmul, primitive, MLP,
  vmap, grad, and train-step cases against deterministic oracle values.

## Phase 4: JAX-Like Program Structure

Goal: make LeanAX examples read like staged pure tensor programs, not raw IR
assembly.

Work:

- Design the first embedded DSL layer over the IR.
- Represent parameter tuples or records.
- Represent pure functions such as `forward params x`.
- Add explicit staging boundaries for `jit`.
- Keep generated IR inspectable.

Exit gate:

- A two-layer MLP forward pass is written as LeanAX DSL code and lowered through
  the existing e2e path.

Current progress:

- `LeanAX/DSL.lean` provides checked dense-layer helpers, and `mlp-forward`
  lowers a two-layer forward pass through the golden, MLIR, and numeric gates.
- The DSL also has a first ReLU activation built from a zero constant, broadcast,
  and `stablehlo.maximum`; `relu-forward` is covered by golden, MLIR, lowering
  manifest, numeric-oracle, and validation-failure checks.

## Phase 5: Batching With `vmap`

Goal: match the JAX habit of writing per-example logic and batching it with a
transform where that is useful.

Work:

- Implement `vmap` for elementwise ops.
- Extend `vmap` to matmul and broadcast patterns needed by dense layers.
- Add shape preservation checks.
- Add at least one preservation theorem for a small pointwise subset.

Exit gate:

- A scalar or per-example expression can be transformed into a batched module
  and checked against a manually batched oracle.

Current progress:

- `LeanAX/Transform.lean` batches a pointwise scalar module by prepending a batch
  dimension, and `vmap-pointwise` is checked numerically.

## Phase 6: Reverse-Mode `grad`

Goal: generate backward programs for scalar losses.

Work:

- Define primitive derivative rules for add, multiply, matmul, broadcast,
  reduce-sum, and relu.
- Represent adjoints in the IR.
- Restrict the first `grad` to scalar losses and static shapes.
- Validate gradients against finite differences or JAX on tiny examples.

Exit gate:

- LeanAX computes gradients for a small dense model loss and matches a Python
  oracle.

Current progress:

- `LeanAX/Grad.lean` covers a restricted scalar-loss case, `grad-square-sum`,
  and the e2e oracle checks that it returns `2 * x`.
- `LeanAX/Loss.lean` adds a first fixed-shape two-class softmax cross-entropy
  loss. It is intentionally not numerically stabilized yet; the current purpose
  is to make classification-loss structure explicit and checked before moving
  to batched MNIST labels.

## Phase 7: Training Step

Goal: lower a JAX-like `train_step`.

Work:

- Represent parameter updates.
- Add host-side parameter storage.
- Compile or run forward, loss, grad, and update kernels.
- Add metrics for loss and accuracy.
- Keep the training step pure at the LeanAX level.

Exit gate:

- A toy dense classifier trains on synthetic data and loss decreases.

Current progress:

- `LeanAX/Training.lean` emits a scalar train-step update module, and
  `e2e/python/training_loop.py` verifies loss reduction on deterministic
  synthetic linear data.

## Phase 8: MNIST MLP

Goal: train a small MLP on MNIST end to end.

Work:

- Add MNIST loading in the host runtime layer.
- Normalize and batch data outside the trusted Lean core.
- Define a two-layer MLP in LeanAX.
- Train for one short epoch.
- Report loss and accuracy.

Exit gate:

- `leanax train mnist-mlp` or an equivalent runner command trains the model,
  prints metrics, and exercises the checked compiler path.

## What Not To Do Yet

- Do not chase a full NumPy surface area.
- Do not make proofs mandatory for every user program.
- Do not implement a custom device compiler.
- Do not hide the IR so deeply that debugging lowering becomes hard.
- Do not optimize performance before the numeric e2e path exists.
