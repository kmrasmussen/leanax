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
- The first external runtime slice is implemented as `affine-runtime`: LeanAX
  emits executable LLVM-dialect MLIR, `mlir-runner` runs it through the local
  LLVM JIT, and the e2e runner compares the scalar checksum against the fixture
  value `94.25`. This does not claim direct StableHLO runtime execution yet.

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
- The first dense-layer batching slice is implemented as `vmap-dense`: it keeps
  weights and bias unbatched, prepends a batch axis to the per-example input,
  and checks the resulting dense module against a manually batched oracle.

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
- `grad-dense-loss` adds the first dense-model gradient artifact for a one-layer
  squared loss. It emits `grad_w = x^T @ (2 * (x @ w + b))` and checks that
  matrix against a Python analytic oracle.

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
- LeanAX modules now support multiple return values, and `sgd-parameter-tree`
  updates a small weight matrix plus bias vector together against a Python
  oracle. This is the first parameter-tree-shaped optimizer artifact.
- `mnist-parameter-tree` expands that update pattern to the full classifier
  parameter set and returns all four updated tensors.

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

Current progress:

- `e2e/python/mnist_fixture.py` provides a deterministic MNIST-shaped fixture
  path with flattened `28x28` image batches, normalized pixel values, and
  one-hot labels. The e2e manifest has a dedicated `data-loader` check for this
  host-side contract.
- `e2e/python/mnist_mlp_smoke.py` is the first documented MNIST MLP smoke
  command. It runs over the fixture path, asserts generated loss and optimizer
  artifacts are present, trains a tiny parity classifier for stable metrics, and
  stays explicit about not being full runtime execution yet.
- `mnist-cross-entropy` now lowers a batched ten-class fixture loss with
  row-wise softmax normalization and mean-over-batch semantics. The e2e manifest
  checks it with golden text, MLIR parsing, lowering manifest validation, and a
  Python numeric oracle.
- `mnist-forward` now lowers a classifier-shaped fixture forward pass with
  inputs `2x784`, hidden width `8`, and logits `2x10`.
- `mnist-parameter-tree` updates the full classifier parameter set
  `w1`, `b1`, `w2`, and `b2` and checks every returned tensor numerically.
- `grad-softmax-dense` computes final-layer gradients for batched ten-class
  softmax cross entropy, including batch-mean scaling.
- `grad-relu-dense` computes first-layer gradients using an explicit ReLU mask
  and returns `grad_w1` plus `grad_b1`.
- `mnist-train-step-artifact` stitches the generated forward, loss, gradient,
  and full parameter-tree artifacts for one fixture batch and checks loss
  reduction plus non-zero updates.
- `mnist-classifier-smoke` trains the ten-class fixture classifier for a short
  deterministic run and asserts loss improves without accuracy regression.

## Where We Are Relative To A Real Classifier

LeanAX is past the project-scaffold stage and has enough checked pieces to make
the remaining MNIST path concrete, but it is not yet training a real MNIST
classifier from LeanAX-generated model and gradient artifacts.

What is already solid:

- The compiler path is reproducible under Nix.
- Generated artifacts have golden text, structural checks, MLIR generic parsing,
  lowering manifests, and Python numeric oracles.
- The DSL can lower small dense layers, ReLU, a two-layer forward pass,
  softmax-style cross entropy, dense batching, first gradients, and parameter
  updates.
- The host side can produce deterministic MNIST-shaped batches and run a stable
  smoke training command.
- One external runtime fixture now runs through LLVM `mlir-runner`.

What is still missing for an honest MNIST classifier:

- The generated classifier artifacts are stitched by an e2e train-step check,
  but not yet lowered as one monolithic Lean train-step module.
- The MNIST classifier smoke command is ten-class and artifact-checked, but it
  is still a Python e2e script rather than a user-facing LeanAX command.
- The IDX parser exists, but full-dataset cache discovery, split selection, and
  opt-in short-epoch metrics remain future work.
- Runtime execution is proven only for a tiny LLVM-dialect affine fixture, not
  for classifier-shaped StableHLO or dense kernels.
- The current Nix shell still lacks a direct StableHLO semantic verifier, so
  generic MLIR parsing and numeric oracles carry most of the generated-artifact
  coverage.

Pragmatically, the project now has enough checked pieces to move from
artifact-composed fixture training toward a cleaner classifier product surface:
first one monolithic train-step artifact, then a user-facing runner, then an
opt-in full-dataset smoke path, and finally direct runtime hardening.

## Phase 9: Ten-Class MNIST Semantics

Goal: make the model and loss match MNIST shapes before adding full training.

Work:

- Add batched ten-class cross entropy with explicit mean loss semantics.
- Add an MNIST-shaped forward module with inputs shaped `batch x 784`, hidden
  dimension kept small enough for readable goldens, and logits shaped
  `batch x 10`.
- Extend validation failures for label/logit rank, class count, and batch
  mismatches.
- Add numeric oracles that check both the generated forward pass and loss
  against deterministic fixture values.

Exit gate:

- `mnist-forward` and `mnist-cross-entropy` are manifested numeric cases with
  golden text, MLIR parsing, lowering manifests, and deterministic oracle
  comparisons.

Current progress:

- `mnist-cross-entropy` covers the batched ten-class loss slice for batch size
  `2`, including a validation-failure case for mismatched label/logit shapes.
- `mnist-forward` covers the classifier-shaped forward slice with a small hidden
  dimension and deterministic oracle values.

## Phase 10: Full Classifier Train Step

Goal: lower the train-step shape used by the classifier.

Work:

- Represent the full parameter tree: `w1`, `b1`, `w2`, and `b2`.
- Add gradient artifacts for cross-entropy logits and the final dense layer.
- Add ReLU gradient gating and chain it through the first dense layer.
- Add a checked SGD update that returns the full updated parameter tree.
- Keep host-side storage outside Lean, but make the update artifact explicit and
  inspectable.

Exit gate:

- A `mnist-train-step` numeric case consumes one fixture batch, returns updated
  parameters, and matches a Python analytic oracle for the same small model.

Current progress:

- `mnist-parameter-tree` covers the full four-output SGD update for classifier
  parameters.
- `grad-softmax-dense` covers the final dense layer gradient for batched
  softmax cross entropy.
- `grad-relu-dense` covers the first dense layer gradient with explicit ReLU
  mask semantics.
- `mnist-train-step-artifact` composes the generated artifacts in the e2e layer
  for one deterministic fixture update.

## Phase 11: Classifier Command And Metrics

Goal: turn the checked artifacts into a user-facing classifier training command.

Work:

- Replace the current parity smoke with a ten-class fixture-mode classifier run.
- Add a command wrapper for `leanax train mnist-mlp` or the closest equivalent
  runner command.
- Print loss, accuracy, sample count, batch count, and artifact paths.
- Add an optional full-dataset route with documented cache location and a fast
  default fixture mode for CI/e2e.
- Keep direct StableHLO/IREE runtime execution as a separate hardening track
  until packaging is stable.

Exit gate:

- One command trains the fixture-mode ten-class classifier, proves loss improves
  or accuracy does not regress, and the full Nix e2e gate covers it.

Current progress:

- `mnist-full-dataset-smoke` runs cached mode against a tiny local IDX cache and
  checks stable metric output without downloading MNIST.
- `mnist-cache-resolver` checks train/test split resolution against a tiny local
  IDX cache, including explicit path resolution and missing-cache diagnostics.
- `mnist_train_command.py --mode fixture` is the current user-facing training
  wrapper. It checks the generated classifier artifacts, runs the deterministic
  fixture classifier, and prints stable metric fields for future parsing.
- `mnist-classifier-smoke` is the current ten-class fixture-mode command. It
  uses checked LeanAX artifacts for the compiler path and host Python for the
  short training loop.
- `mnist-progress-report` is the e2e-produced readiness report. It inspects the
  manifest and generated artifacts, prints stable milestone booleans, and fails
  if the report no longer matches this roadmap state.
- The report currently marks the IDX loader and fixture ten-class training as
  present, while keeping full-dataset training, a monolithic MNIST train-step
  artifact, and direct MNIST external-runtime execution as future work.

## Phase 12: Monolithic Classifier Artifact

Goal: reduce the current host-composed train step into one fixed-shape LeanAX
artifact that owns the classifier update contract.

Work:

- Add a fixed `mnist-train-step` module with inputs for one `2x784` batch,
  one-hot labels, explicit `relu_mask`, `w1`, `b1`, `w2`, and `b2`.
- Return updated `w1`, `b1`, `w2`, `b2`, and enough loss/logit information for
  a deterministic oracle to prove the update.
- Reuse the existing explicit ReLU mask semantics until compare/select support
  is strong enough to derive the mask inside the artifact.
- Add a numeric oracle that compares the monolithic artifact with the current
  artifact-composed Python train step.
- Add validation failures for mismatched batch, class, hidden, and parameter
  tree shapes.

Exit gate:

- `mnist-train-step` is a manifested numeric case with golden text, lowering
  manifest validation, MLIR parsing, and oracle coverage. The progress report
  flips `monolithic_mnist_train_step` to true in the same commit.

Current progress:

- `mnist-forward-runtime` runs a small dense-ReLU-dense forward checksum through
  LLVM `mlir-runner` as a documented stepping stone toward full classifier
  runtime execution.
- `dense-runtime` runs a fixed dense-layer checksum through LLVM `mlir-runner`,
  expanding runtime coverage beyond the affine fixture while remaining separate
  from full MNIST runtime execution.
- `runtime-capability-matrix` reports required LLVM/MLIR runtime tools and
  optional direct StableHLO/IREE tooling from inside the Nix shell.
- `mnist-train-step` now lowers the fixed `2x784 -> 8 -> 10` classifier update
  as one module. It returns `next_w1`, `next_b1`, `next_w2`, `next_b2`, and the
  batch loss, and the numeric oracle compares it against the previous composed
  train-step math.
- The first version keeps `relu_mask` explicit, matching the existing
  first-layer gradient artifact until comparison/select primitives can derive
  the mask inside the module.
- The manifest includes expected failures for label/class mismatch, hidden mask
  mismatch, and parameter update mismatch.

Tickets:

- `TICKET-0036`: Monolithic MNIST Train-Step Artifact.
- `TICKET-0037`: MNIST Train-Step Shape Validation Suite.

## Phase 13: Product Runner And Full-Dataset Smoke

Goal: make the classifier path usable from one command while keeping default CI
fast, hermetic, and fixture-based.

Work:

- Add a user-facing runner command or wrapper for fixture-mode
  `leanax train mnist-mlp`.
- Print metrics with stable field names: mode, samples, batches, epochs,
  first/final loss, first/final accuracy, and artifact paths.
- Resolve full MNIST IDX files from explicit paths or the documented cache
  directory without downloading in the default gate.
- Add an opt-in full-dataset smoke that can run a short epoch when the cache is
  present and skip with a clear diagnostic when it is absent.
- Keep the e2e manifest network-free by testing cache resolution against tiny
  local IDX files and fixture mode by default.

Exit gate:

- One command runs the fixture classifier through the checked compiler path and
  prints stable metrics. A separate opt-in command can use cached MNIST IDX
  files and report short-run metrics without changing the default Nix e2e gate.

Tickets:

- `TICKET-0038`: MNIST Train Command Wrapper.
- `TICKET-0039`: MNIST Cache Resolver And Split Loader.
- `TICKET-0040`: Optional Full-Dataset Metrics Smoke.

## Phase 14: Runtime Hardening

Goal: move runtime execution from the tiny affine fixture toward classifier
shaped kernels without pretending the StableHLO runtime story is solved.

Work:

- Record a runtime capability matrix for the current Nix shell: generic MLIR
  parsing, optional `stablehlo-opt`, LLVM `mlir-runner`, and any available IREE
  or StableHLO execution route.
- Add a dense or ReLU runtime fixture that exercises more of the classifier op
  mix than `affine-runtime`.
- Add a direct classifier-forward runtime slice only when the available runtime
  path can execute it deterministically.
- Keep fallback diagnostics explicit when direct StableHLO runtime tooling is
  missing.

Exit gate:

- Runtime coverage expands beyond the affine checksum, and the progress report
  distinguishes dense-kernel runtime coverage from full MNIST runtime execution.

Tickets:

- `TICKET-0041`: Runtime Capability Matrix.
- `TICKET-0042`: Dense Kernel Runtime Fixture.
- `TICKET-0043`: MNIST Forward Runtime Slice.

## Phase 15: Progress Report Closeout

Goal: keep the roadmap, ticket queue, and e2e readiness report synchronized as
the next classifier phase lands.

Work:

- Extend `mnist-progress-report` with booleans for monolithic train step,
  command wrapper, full-dataset smoke, dense runtime, and MNIST-forward runtime.
- Link the report output to the phase/ticket table in this roadmap.
- Add a closeout audit note that lists which future milestones remain false and
  why.

Exit gate:

- The readiness report, roadmap, and ticket statuses agree after the next phase
  of classifier work.

Current progress:

- `mnist-progress-report` now covers every ticket in the `TICKET-0036` through
  `TICKET-0044` phase. It marks the monolithic train step, command wrapper,
  cache resolver, optional full-dataset smoke, runtime capability matrix, dense
  runtime, and MNIST-forward runtime as true.
- The same report keeps full real-dataset training and direct full MNIST
  external-runtime execution false, because those are still harder follow-up
  milestones rather than proven default-gate capabilities.

Tickets:

- `TICKET-0044`: Classifier Readiness Report V2.

## Phase 16: Derived ReLU Mask And Cleaner Train Step

Goal: remove the explicit ReLU-mask input from the classifier train-step path by
adding the smallest comparison/select surface needed to derive masks inside
LeanAX IR.

Work:

- Add checked comparison and select primitives to the IR, renderer, verifier,
  numeric oracle, and validation failures.
- Add a small derived ReLU-mask artifact that returns both activated values and
  an f32-compatible mask.
- Add a derived-mask MNIST train-step artifact that accepts the batch, labels,
  and parameter tree but no `relu_mask` input.
- Keep the existing explicit-mask train step as a compatibility and debugging
  fixture until the derived-mask path has enough soak time.
- Extend the readiness report so the mask-derivation milestone cannot drift.

Exit gate:

- `compare-select`, `relu-derived-mask`, and
  `mnist-train-step-derived-mask` are manifested numeric cases with golden text,
  lowering manifests, MLIR parsing, and Python oracle coverage. The progress
  report marks derived ReLU mask and derived-mask train step as true.

Current progress:

- `compare-select` is now a manifested numeric case. LeanAX emits
  `stablehlo.compare` with a predicate tensor result and feeds that predicate to
  `stablehlo.select`; the golden text, lowering manifest, MLIR parse, structural
  verifier, and Python oracle all cover the path.
- `relu-derived-mask` now derives both ReLU activations and an f32 mask from a
  `2x8` hidden pre-activation tensor. This proves the mask can come from LeanAX
  IR instead of from host-side Python.
- `mnist-train-step-derived-mask` now exposes the classifier train-step contract
  without a `relu_mask` input. The mask is computed inside the module and the
  progress report marks the derived-mask train-step milestone true.
- Compare/select validation now has expected-failure coverage for mismatched
  compare operands and mismatched select predicate shape.
- `mnist-progress-report` now covers the full phase-16 surface: compare/select
  artifact support, derived ReLU mask artifact, derived-mask train step, and
  compare/select validation failures. The explicit-mask train step remains as a
  compatibility fixture.

Tickets:

- `TICKET-0045`: Compare And Select Primitives.
- `TICKET-0046`: Derived ReLU Mask Artifact.
- `TICKET-0047`: MNIST Train Step With Derived Mask.
- `TICKET-0048`: Derived Mask Validation Suite.
- `TICKET-0049`: Classifier Readiness Report V3.

## Phase 17: Derived Command Path And Real Dataset Training

Goal: move the command-facing classifier path onto the derived-mask train-step
and turn cached IDX MNIST training from a tiny smoke into an explicit,
reportable real-dataset milestone.

Work:

- Update the train command and artifact composition checks so the derived-mask
  train-step is the preferred generated update artifact.
- Keep the explicit-mask train-step as a compatibility fixture, but stop making
  command-facing training depend on a host-provided ReLU mask.
- Add a cached real-dataset training sweep that uses actual IDX samples from the
  resolver, not the tiny fixture batch, with bounded default-gate runtime.
- Emit stable machine-readable metrics for dataset size, sample count, loss,
  accuracy, and artifact paths.
- Extend the readiness report so derived command wiring and cached real-dataset
  training cannot drift.
- Move as much regression checking and training plumbing as practical into the
  Rust e2e harness over time. Python should remain mostly a reference and
  comparison layer, especially where it mirrors JAX/NumPy behavior or gives a
  compact oracle. Keep this balanced: do not force Rust rewrites when Python is
  still the clearest way to express reference math or inspect JAX-adjacent
  behavior.

Exit gate:

- The default e2e manifest includes derived-mask command coverage and a cached
  real-dataset training sweep. `mnist-progress-report` distinguishes fixture
  training, cached real-dataset training, and still-unproven direct full MNIST
  external runtime.

Tickets:

- `TICKET-0050`: Derived Mask Train Command Wiring.
- `TICKET-0051`: Cached Real Dataset Training Sweep.
- `TICKET-0052`: Dataset Training Metrics Artifact.
- `TICKET-0053`: Classifier Readiness Report V4.
- `TICKET-0054`: Direct MNIST Runtime Boundary Plan.

Current progress:

- `mnist_train_command.py` now uses and reports
  `generated/mnist-train-step-derived-mask.mlir` as its command-facing
  train-step artifact. The explicit-mask artifact remains a compatibility
  numeric case.
- `mnist-cached-training-sweep` now exercises a bounded train-split IDX cache
  with sixteen samples through the resolver, not the embedded fixture batch.
  It proves loss improvement and non-regressing accuracy in the default gate.
- `generated/mnist-real-dataset-metrics.json` records stable cached training
  metrics with schema `leanax.mnist_dataset_metrics.v1`, and
  `mnist-dataset-metrics` verifies the schema, values, and generated artifact
  references immediately after the sweep.
- `mnist-progress-report` now runs after the cached sweep and metrics verifier.
  It marks derived-mask command wiring, cached dataset training, structured
  dataset metrics, and `full_dataset_training` true while keeping direct MNIST
  external runtime false.

## Phase 18: Direct Runtime Boundary

Goal: make the next runtime hardening slice concrete without depending on
unpackaged StableHLO or IREE tooling.

Work:

- Document the direct MNIST runtime boundary and what would count as a real
  external-runtime classifier milestone.
- Use the current runtime capability matrix and
  `mnist-train-step-derived-mask` artifacts as the evidence base.
- Prefer expanding the existing LLVM `mlir-runner` route for the next slice,
  because required MLIR runtime tools are available in the Nix shell while
  optional StableHLO and IREE tools are not.
- Add a verifier that inventories the operations and shapes required by the
  derived-mask train-step artifact.
- Add focused LLVM runtime fixtures for scalar math and then a tiny
  derived-mask train-step checksum before attempting the full fixed
  `2x784 -> 8 -> 10` runtime artifact.
- Extend the readiness report only after the runtime fixture is actually
  manifested.

Exit gate:

- The runtime plan names the chosen route, rejects premature runtime routes, and
  breaks the next implementation slice into concrete tickets.

Tickets:

- `TICKET-0054`: Direct MNIST Runtime Boundary Plan.
- `TICKET-0055`: Runtime Operation Inventory Verifier.
- `TICKET-0056`: Runtime Scalar Math Fixture.
- `TICKET-0057`: Tiny Derived-Mask Train-Step Runtime Fixture.
- `TICKET-0058`: Runtime Readiness Report V5.

Current progress:

- `docs/mnist-runtime-boundary.md` chooses LLVM lowering expansion through
  `mlir-runner` for the next slice, because the current Nix shell has
  `mlir-opt` and `mlir-runner` but not `stablehlo-opt`, `iree-compile`, or
  `iree-run-module`.
- The plan maps the classifier train-step inputs, outputs, intermediate tensor
  shapes, and required StableHLO-shaped operations from
  `mnist-train-step-derived-mask`.
- The plan keeps `direct_mnist_external_runtime` false until an external
  runtime artifact executes the train-step semantics and compares loss plus
  updated parameter checksums against the existing oracle.
- `runtime-operation-inventory` now verifies the derived-mask train-step
  lowering manifest in the default e2e gate and prints the unsupported runtime
  operation surface for the next LLVM expansion slice.
- `softmax-loss-runtime` now executes a scalar softmax cross-entropy checksum
  with LLVM exp/log intrinsics and floating-point division, proving the scalar
  math route needed by the train-step loss before expanding to tensor-shaped
  runtime code.

## What Not To Do Yet

- Do not chase a full NumPy surface area.
- Do not make proofs mandatory for every user program.
- Do not implement a custom device compiler.
- Do not hide the IR so deeply that debugging lowering becomes hard.
- Do not optimize performance before the numeric e2e path exists.
- Do not let Python become the permanent owner of regression orchestration or
  training plumbing when the Rust harness can reasonably own that behavior.
