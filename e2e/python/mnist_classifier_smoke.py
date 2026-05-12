from __future__ import annotations

from mnist_fixture import fixture_batches
from mnist_train_step_artifact import (
    CLASSES,
    HIDDEN,
    LR,
    column_sum,
    forward,
    loss,
    matmul,
    patterned,
    patterned_vector,
    require_artifact,
    softmax,
    sub_scaled,
    sub_scaled_vector,
    transpose,
)


EPOCHS = 8


def accuracy(logits: list[list[float]], labels: list[list[float]]) -> float:
    correct = 0
    total = 0
    for logit_row, label_row in zip(logits, labels):
        predicted = max(range(len(logit_row)), key=lambda index: logit_row[index])
        expected = label_row.index(1.0)
        correct += 1 if predicted == expected else 0
        total += 1
    return correct / total


def evaluate(
    batches,
    w1: list[list[float]],
    b1: list[float],
    w2: list[list[float]],
    b2: list[float],
) -> tuple[float, float]:
    total_loss = 0.0
    total_accuracy = 0.0
    total_batches = 0
    for batch in batches:
        _hidden, _mask, logits = forward(batch.images, w1, b1, w2, b2)
        total_loss += loss(logits, batch.labels)
        total_accuracy += accuracy(logits, batch.labels)
        total_batches += 1
    return total_loss / total_batches, total_accuracy / total_batches


def apply_batch(
    images: list[list[float]],
    labels: list[list[float]],
    w1: list[list[float]],
    b1: list[float],
    w2: list[list[float]],
    b2: list[float],
) -> tuple[list[list[float]], list[float], list[list[float]], list[float]]:
    hidden, mask, logits = forward(images, w1, b1, w2, b2)
    probs = softmax(logits)
    scale = 1.0 / len(images)
    logit_delta = [
        [(prob - label) * scale for prob, label in zip(prob_row, label_row)]
        for prob_row, label_row in zip(probs, labels)
    ]
    grad_w2 = matmul(transpose(hidden), logit_delta)
    grad_b2 = column_sum(logit_delta)
    hidden_grad = matmul(logit_delta, transpose(w2))
    pre_activation_grad = [
        [grad * mask_value for grad, mask_value in zip(grad_row, mask_row)]
        for grad_row, mask_row in zip(hidden_grad, mask)
    ]
    grad_w1 = matmul(transpose(images), pre_activation_grad)
    grad_b1 = column_sum(pre_activation_grad)
    return (
        sub_scaled(w1, grad_w1, LR),
        sub_scaled_vector(b1, grad_b1, LR),
        sub_scaled(w2, grad_w2, LR),
        sub_scaled_vector(b2, grad_b2, LR),
    )


def train_batches(batches, epochs: int = EPOCHS) -> tuple[float, float, float, float]:
    w1 = patterned(28 * 28, HIDDEN, 0.002, 3)
    b1 = patterned_vector(HIDDEN, 0.01, 5)
    w2 = patterned(HIDDEN, CLASSES, 0.02, 7)
    b2 = patterned_vector(CLASSES, 0.01, 9)
    first_loss, first_accuracy = evaluate(batches, w1, b1, w2, b2)
    for _epoch in range(epochs):
        for batch in batches:
            w1, b1, w2, b2 = apply_batch(batch.images, batch.labels, w1, b1, w2, b2)
    final_loss, final_accuracy = evaluate(batches, w1, b1, w2, b2)
    return first_loss, final_loss, first_accuracy, final_accuracy


def train() -> tuple[float, float, float, float]:
    return train_batches(fixture_batches(), EPOCHS)


def main() -> int:
    require_artifact("generated/mnist-forward.mlir", ["tensor<2x784xf32>", "tensor<2x10xf32>"])
    require_artifact("generated/mnist-cross-entropy.mlir", ["stablehlo.exponential", "value = \"-0.5\""])
    require_artifact("generated/grad-softmax-dense.mlir", ["%grad_w2", "%grad_b2"])
    require_artifact("generated/grad-relu-dense.mlir", ["%grad_w1", "%grad_b1"])
    require_artifact("generated/mnist-parameter-tree.mlir", ["%next_w1", "%next_b1", "%next_w2", "%next_b2"])

    first_loss, final_loss, first_accuracy, final_accuracy = train()
    if not final_loss < first_loss:
        raise AssertionError(f"classifier loss did not decrease: {first_loss} -> {final_loss}")
    if final_accuracy < first_accuracy:
        raise AssertionError(f"classifier accuracy regressed: {first_accuracy} -> {final_accuracy}")
    print(
        "mnist-classifier-smoke "
        f"first_loss={first_loss:.6f} final_loss={final_loss:.6f} "
        f"first_accuracy={first_accuracy:.2f} final_accuracy={final_accuracy:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
