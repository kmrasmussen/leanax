from __future__ import annotations

import math
from pathlib import Path

from mnist_fixture import fixture_batches


HIDDEN = 8
CLASSES = 10
LR = 0.2


def require_artifact(path: str, required_text: list[str]) -> None:
    text = Path(path).read_text(encoding="utf-8")
    for required in required_text:
        if required not in text:
            raise AssertionError(f"{path} does not contain {required}")


def patterned(rows: int, cols: int, scale: float, offset: int) -> list[list[float]]:
    return [
        [(((row * cols + col + offset) % 17) - 8) * scale for col in range(cols)]
        for row in range(rows)
    ]


def patterned_vector(size: int, scale: float, offset: int) -> list[float]:
    return [(((index + offset) % 17) - 8) * scale for index in range(size)]


def matmul(lhs: list[list[float]], rhs: list[list[float]]) -> list[list[float]]:
    return [
        [sum(lhs[row][inner] * rhs[inner][col] for inner in range(len(rhs))) for col in range(len(rhs[0]))]
        for row in range(len(lhs))
    ]


def add_bias(values: list[list[float]], bias: list[float]) -> list[list[float]]:
    return [[value + bias[col] for col, value in enumerate(row)] for row in values]


def relu(values: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    activated = [[max(value, 0.0) for value in row] for row in values]
    mask = [[1.0 if value > 0.0 else 0.0 for value in row] for row in values]
    return activated, mask


def softmax(values: list[list[float]]) -> list[list[float]]:
    out = []
    for row in values:
        shifted = [value - max(row) for value in row]
        exps = [math.exp(value) for value in shifted]
        denom = sum(exps)
        out.append([value / denom for value in exps])
    return out


def transpose(values: list[list[float]]) -> list[list[float]]:
    return [[values[row][col] for row in range(len(values))] for col in range(len(values[0]))]


def column_sum(values: list[list[float]]) -> list[float]:
    return [sum(row[col] for row in values) for col in range(len(values[0]))]


def sub_scaled(values: list[list[float]], grads: list[list[float]], lr: float) -> list[list[float]]:
    return [
        [value - lr * grads[row][col] for col, value in enumerate(row_values)]
        for row, row_values in enumerate(values)
    ]


def sub_scaled_vector(values: list[float], grads: list[float], lr: float) -> list[float]:
    return [value - lr * grads[index] for index, value in enumerate(values)]


def forward(
    images: list[list[float]],
    w1: list[list[float]],
    b1: list[float],
    w2: list[list[float]],
    b2: list[float],
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    hidden_pre = add_bias(matmul(images, w1), b1)
    hidden, mask = relu(hidden_pre)
    logits = add_bias(matmul(hidden, w2), b2)
    return hidden, mask, logits


def loss(logits: list[list[float]], labels: list[list[float]]) -> float:
    probs = softmax(logits)
    total = 0.0
    for prob_row, label_row in zip(probs, labels):
        for prob, label in zip(prob_row, label_row):
            if label == 1.0:
                total -= math.log(prob)
    return total / len(logits)


def train_step() -> tuple[float, float, float]:
    batch = fixture_batches()[0]
    w1 = patterned(28 * 28, HIDDEN, 0.002, 3)
    b1 = patterned_vector(HIDDEN, 0.01, 5)
    w2 = patterned(HIDDEN, CLASSES, 0.02, 7)
    b2 = patterned_vector(CLASSES, 0.01, 9)

    hidden, mask, logits = forward(batch.images, w1, b1, w2, b2)
    first_loss = loss(logits, batch.labels)
    probs = softmax(logits)
    scale = 1.0 / len(batch.images)
    logit_delta = [
        [(prob - label) * scale for prob, label in zip(prob_row, label_row)]
        for prob_row, label_row in zip(probs, batch.labels)
    ]
    grad_w2 = matmul(transpose(hidden), logit_delta)
    grad_b2 = column_sum(logit_delta)
    hidden_grad = matmul(logit_delta, transpose(w2))
    pre_activation_grad = [
        [grad * mask_value for grad, mask_value in zip(grad_row, mask_row)]
        for grad_row, mask_row in zip(hidden_grad, mask)
    ]
    grad_w1 = matmul(transpose(batch.images), pre_activation_grad)
    grad_b1 = column_sum(pre_activation_grad)

    next_w1 = sub_scaled(w1, grad_w1, LR)
    next_b1 = sub_scaled_vector(b1, grad_b1, LR)
    next_w2 = sub_scaled(w2, grad_w2, LR)
    next_b2 = sub_scaled_vector(b2, grad_b2, LR)
    _hidden, _mask, next_logits = forward(batch.images, next_w1, next_b1, next_w2, next_b2)
    final_loss = loss(next_logits, batch.labels)
    update_l1 = (
        sum(abs(a - b) for row_a, row_b in zip(next_w1, w1) for a, b in zip(row_a, row_b))
        + sum(abs(a - b) for a, b in zip(next_b1, b1))
        + sum(abs(a - b) for row_a, row_b in zip(next_w2, w2) for a, b in zip(row_a, row_b))
        + sum(abs(a - b) for a, b in zip(next_b2, b2))
    )
    return first_loss, final_loss, update_l1


def main() -> int:
    require_artifact("generated/mnist-forward.mlir", ["tensor<2x784xf32>", "stablehlo.maximum", "tensor<2x10xf32>"])
    require_artifact("generated/mnist-cross-entropy.mlir", ["stablehlo.exponential", "value = \"-0.5\""])
    require_artifact("generated/grad-softmax-dense.mlir", ["%grad_w2", "%grad_b2"])
    require_artifact("generated/grad-relu-dense.mlir", ["%grad_w1", "%grad_b1", "%relu_mask"])
    require_artifact("generated/mnist-parameter-tree.mlir", ["%next_w1", "%next_b1", "%next_w2", "%next_b2"])

    first_loss, final_loss, update_l1 = train_step()
    if not final_loss < first_loss:
        raise AssertionError(f"train step did not reduce fixture loss: {first_loss} -> {final_loss}")
    if not update_l1 > 0.0:
        raise AssertionError("train step produced no parameter update")
    print(
        "mnist-train-step-artifact "
        f"first_loss={first_loss:.6f} final_loss={final_loss:.6f} update_l1={update_l1:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
