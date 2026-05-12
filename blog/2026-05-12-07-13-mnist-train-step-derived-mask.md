# MNIST Train Step With Derived Mask

`TICKET-0047` adds the cleaner train-step contract.

`mnist-train-step-derived-mask` takes images, labels, `w1`, `b1`, `w2`, and
`b2`. It computes hidden pre-activations, derives the ReLU predicate and f32
mask internally, and returns the same updated parameter tree plus loss as the
explicit-mask fixture.

The old `mnist-train-step` remains in the manifest as a compatibility fixture,
but the new artifact removes the host-provided mask edge from the main
classifier update path.
