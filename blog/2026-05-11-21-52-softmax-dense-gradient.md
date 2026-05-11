# Softmax Dense Gradient

`TICKET-0030` is complete.

LeanAX now emits the final-layer classifier gradient as `grad-softmax-dense`.
The artifact takes hidden activations, logits, and one-hot labels, computes the
batched softmax delta `(probs - labels) / batch`, and returns gradients for
`w2` and `b2`.

The e2e path checks golden text, MLIR generic parsing, lowering manifest
validation, and a Python analytic oracle. The new validation-failure case catches
mismatched logits and labels before the gradient can be lowered.
