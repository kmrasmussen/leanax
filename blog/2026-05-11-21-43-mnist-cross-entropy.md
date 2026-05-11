# MNIST Cross Entropy

`TICKET-0027` is complete.

LeanAX now has a batched ten-class cross-entropy artifact for the MNIST fixture
shape. The new `mnist-cross-entropy` case takes `2x10` logits and `2x10`
one-hot labels, computes row-wise softmax probabilities, reduces the weighted
log probabilities per example, and returns the mean scalar loss.

The small IR addition is `reduceSumLastDim`, which keeps a singleton final axis.
That gives the loss a row-wise denominator shaped `2x1`, which can broadcast
back to `2x10` logits. The e2e manifest checks the generated module with golden
text, MLIR generic parsing, lowering manifest validation, and a deterministic
Python numeric oracle. It also adds an expected failure for mismatched MNIST
label/logit shapes.
