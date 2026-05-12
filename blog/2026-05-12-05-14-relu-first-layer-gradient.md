# ReLU First-Layer Gradient

`TICKET-0031` is complete.

LeanAX now has `grad-relu-dense`, the first-layer gradient artifact for the
MNIST-shaped classifier. It takes flattened images, hidden-layer gradients, and
an explicit ReLU mask. The artifact multiplies the incoming hidden gradient by
the mask, computes `grad_w1 = x^T @ pre_activation_grad`, and reduces across the
batch for `grad_b1`.

The mask is deliberately explicit in this slice because LeanAX does not yet have
comparison/select primitives for deriving it from pre-activations. That keeps
the ReLU derivative semantics visible while still giving the train-step path a
checked first-layer gradient artifact. The e2e runner checks golden text, MLIR
parsing, lowering manifest validation, a Python analytic oracle, and an
expected failure for hidden-gradient/mask shape mismatch.
