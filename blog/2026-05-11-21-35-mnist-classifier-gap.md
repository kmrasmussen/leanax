# MNIST Classifier Gap

The roadmap now has a clearer boundary between the current smoke path and real
MNIST classifier training.

LeanAX has the compiler/e2e skeleton, MNIST-shaped fixture batches, small MLP
forward artifacts, first loss and gradient artifacts, a parameter update path,
and one external runtime slice. The missing part is the full classifier shape:
batched ten-class loss, `784 -> hidden -> 10` forward, gradients through
softmax/ReLU/two dense layers, a four-parameter update, and a command that trains
that checked artifact path instead of a tiny host-side parity model.

The new ticket queue starts at `TICKET-0027` and ends at `TICKET-0035`. It moves
from ten-class loss and MNIST-shaped forward artifacts toward a checked train
step, fixture-mode classifier command, optional full-dataset route, and an e2e
progress report that keeps the project honest about what is actually proven.
