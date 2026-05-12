from __future__ import annotations

from dataclasses import dataclass
import math
import re
import sys
from pathlib import Path


@dataclass(frozen=True)
class Tensor:
    shape: tuple[int, ...]
    data: tuple[float, ...]

    @staticmethod
    def scalar(value: float) -> "Tensor":
        return Tensor((), (value,))


def fail(message: str) -> None:
    print(f"numeric oracle failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def numel(shape: tuple[int, ...]) -> int:
    out = 1
    for dim in shape:
        out *= dim
    return out


def parse_type(text: str) -> tuple[int, ...]:
    body = text.removeprefix("tensor<").removesuffix(">")
    fields = body.split("x")
    if len(fields) == 1:
        return ()
    return tuple(int(field) for field in fields[:-1])


def flat_index(indices: tuple[int, ...], shape: tuple[int, ...]) -> int:
    offset = 0
    stride = 1
    for index, dim in zip(reversed(indices), reversed(shape)):
        offset += index * stride
        stride *= dim
    return offset


def unflatten(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    coords = []
    for dim in reversed(shape):
        coords.append(index % dim)
        index //= dim
    return tuple(reversed(coords))


def elementwise(lhs: Tensor, rhs: Tensor, op) -> Tensor:
    if lhs.shape != rhs.shape:
        fail(f"shape mismatch: {lhs.shape} vs {rhs.shape}")
    return Tensor(lhs.shape, tuple(op(a, b) for a, b in zip(lhs.data, rhs.data)))


def broadcast_to(value: Tensor, shape: tuple[int, ...]) -> Tensor:
    if not value.shape:
        return Tensor(shape, tuple(value.data[0] for _ in range(numel(shape))))
    if len(value.shape) > len(shape):
        fail(f"cannot broadcast {value.shape} to {shape}")
    offset = len(shape) - len(value.shape)
    for source_dim, target_dim in zip(value.shape, shape[offset:]):
        if source_dim != target_dim and source_dim != 1:
            fail(f"cannot broadcast {value.shape} to {shape}")

    out = []
    for index in range(numel(shape)):
        coords = unflatten(index, shape)
        operand_coords = tuple(
            0 if dim == 1 else coord
            for coord, dim in zip(coords[offset:], value.shape)
        )
        out.append(value.data[flat_index(operand_coords, value.shape)])
    return Tensor(shape, tuple(out))


def matmul(lhs: Tensor, rhs: Tensor) -> Tensor:
    if len(lhs.shape) != 2 or len(rhs.shape) != 2 or lhs.shape[1] != rhs.shape[0]:
        fail(f"bad matmul shapes: {lhs.shape} and {rhs.shape}")
    rows, inner = lhs.shape
    _, cols = rhs.shape
    out = []
    for row in range(rows):
        for col in range(cols):
            total = 0.0
            for k in range(inner):
                total += lhs.data[row * inner + k] * rhs.data[k * cols + col]
            out.append(total)
    return Tensor((rows, cols), tuple(out))


def transpose_2d(value: Tensor) -> Tensor:
    if len(value.shape) != 2:
        fail(f"expected rank-2 transpose, got {value.shape}")
    rows, cols = value.shape
    return Tensor((cols, rows), tuple(value.data[r * cols + c] for c in range(cols) for r in range(rows)))


def reduce_last_dim(value: Tensor) -> Tensor:
    if len(value.shape) != 2:
        fail(f"expected rank-2 reduce-last-dim, got {value.shape}")
    rows, cols = value.shape
    return Tensor((rows, 1), tuple(sum(value.data[row * cols + col] for col in range(cols)) for row in range(rows)))


def parse_signature(text: str) -> list[str]:
    signature = re.search(r"func\.func @main\((.*?)\) ->", text, re.S)
    if signature is None:
        fail("missing function signature")
    return re.findall(r"%([A-Za-z0-9_]+): tensor<[^>]+>", signature.group(1))


def parse_returns(text: str) -> list[str]:
    match = re.search(r"return (.*?) :", text)
    if match is None:
        fail("missing return")
    return re.findall(r"%([A-Za-z0-9_]+)", match.group(1))


def result_shape(line: str) -> tuple[int, ...]:
    matches = re.findall(r"-> (tensor<[^>]+>)", line)
    if not matches:
        fail(f"missing result type in line: {line}")
    return parse_type(matches[-1])


def execute(text: str, inputs: dict[str, Tensor]) -> list[Tensor]:
    values = dict(inputs)
    for name in parse_signature(text):
        if name not in values:
            fail(f"oracle did not provide input %{name}")

    for line in text.splitlines():
        op_match = re.search(r'%([A-Za-z0-9_]+) = "stablehlo\.([A-Za-z0-9_]+)"\(([^)]*)\)', line)
        if op_match is None:
            continue
        name, op, operand_text = op_match.groups()
        operands = re.findall(r"%([A-Za-z0-9_]+)", operand_text)
        tensors = [values[operand] for operand in operands]
        shape = result_shape(line)

        if op == "constant":
            value_match = re.search(r'value = "([^"]+)"', line)
            if value_match is None:
                fail(f"constant %{name} has no value")
            scalar = float(value_match.group(1))
            values[name] = Tensor(shape, tuple(scalar for _ in range(numel(shape))))
        elif op == "add":
            values[name] = elementwise(tensors[0], tensors[1], lambda a, b: a + b)
        elif op == "multiply":
            values[name] = elementwise(tensors[0], tensors[1], lambda a, b: a * b)
        elif op == "maximum":
            values[name] = elementwise(tensors[0], tensors[1], max)
        elif op == "divide":
            values[name] = elementwise(tensors[0], tensors[1], lambda a, b: a / b)
        elif op == "exponential":
            values[name] = Tensor(tensors[0].shape, tuple(math.exp(value) for value in tensors[0].data))
        elif op == "log":
            values[name] = Tensor(tensors[0].shape, tuple(math.log(value) for value in tensors[0].data))
        elif op == "dot_general":
            values[name] = matmul(tensors[0], tensors[1])
        elif op == "broadcast_in_dim":
            values[name] = broadcast_to(tensors[0], shape)
        elif op == "reshape":
            if numel(tensors[0].shape) != numel(shape):
                fail(f"bad reshape from {tensors[0].shape} to {shape}")
            values[name] = Tensor(shape, tensors[0].data)
        elif op == "transpose":
            values[name] = transpose_2d(tensors[0])
        elif op == "reduce":
            if 'dimensions = "last"' in line:
                values[name] = reduce_last_dim(tensors[0])
            else:
                values[name] = Tensor.scalar(sum(tensors[0].data))
        else:
            fail(f"unsupported op stablehlo.{op}")

        if values[name].shape != shape:
            fail(f"%{name} evaluated to {values[name].shape}, expected {shape}")

    return [values[name] for name in parse_returns(text)]


def tensor(shape: tuple[int, ...], data: list[float]) -> Tensor:
    if len(data) != numel(shape):
        fail(f"bad oracle tensor for {shape}: {data}")
    return Tensor(shape, tuple(float(value) for value in data))


def patterned_tensor(shape: tuple[int, ...], scale: float, offset: int = 0) -> Tensor:
    values = [(((index + offset) % 17) - 8) * scale for index in range(numel(shape))]
    return Tensor(shape, tuple(values))


def oracle_inputs(name: str) -> dict[str, Tensor]:
    match name:
        case "affine":
            return {
                "x": tensor((2, 3), [1, 2, 3, 4, 5, 6]),
                "bias": tensor((2, 3), [0.5, -1, 2, 1, 0, -2]),
            }
        case "matmul":
            return {
                "lhs": tensor((2, 4), [1, 2, 3, 4, -1, 0, 2, 1]),
                "rhs": tensor((4, 3), [1, 0, 2, 0, 1, -1, 2, 1, 0, -1, 2, 1]),
            }
        case "nn-primitives":
            return {
                "x": tensor((2, 3), [1, 2, 3, 4, 5, 6]),
                "bias": tensor((3,), [0.25, -0.5, 1.0]),
            }
        case "mlp-forward":
            return {
                "x": tensor((2, 4), [1, 0, -1, 2, 0.5, 1.5, -0.5, 1]),
                "w1": tensor((4, 3), [1, 0, 2, -1, 1, 0, 0.5, -0.5, 1, 2, 1, -1]),
                "b1": tensor((3,), [0.25, -0.25, 0.5]),
                "w2": tensor((3, 2), [1, -1, 0.5, 2, -0.5, 1]),
                "b2": tensor((2,), [0.1, -0.2]),
            }
        case "relu-forward":
            return {
                "x": tensor((2, 4), [1, -2, 0.5, 3, -1, 2, -0.5, 0]),
                "w": tensor((4, 3), [1, -1, 0.5, 2, 0, -0.5, -1, 1.5, 2, 0.25, -2, 1]),
                "b": tensor((3,), [-1, 0.5, 2]),
            }
        case "mnist-forward":
            return {
                "x": patterned_tensor((2, 784), 0.01, 1),
                "w1": patterned_tensor((784, 8), 0.002, 3),
                "b1": patterned_tensor((8,), 0.01, 5),
                "w2": patterned_tensor((8, 10), 0.02, 7),
                "b2": patterned_tensor((10,), 0.01, 9),
            }
        case "cross-entropy-loss":
            return {
                "logits": tensor((2,), [1.25, -0.75]),
                "labels": tensor((2,), [1.0, 0.0]),
            }
        case "mnist-cross-entropy":
            return {
                "logits": tensor(
                    (2, 10),
                    [
                        1.5, -0.25, 0.75, -1.0, 0.5, 0.0, -0.5, 1.0, -1.5, 0.25,
                        -0.75, 0.2, 1.25, 0.0, -0.4, 0.6, -1.2, 1.4, 0.3, -0.1,
                    ],
                ),
                "labels": tensor(
                    (2, 10),
                    [
                        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                    ],
                ),
            }
        case "vmap-pointwise":
            return {
                "x": tensor((4,), [1, 2, -1, 0.5]),
                "y": tensor((4,), [0.5, -3, 4, 1.5]),
            }
        case "vmap-dense":
            return {
                "x": tensor((2, 4), [1, -2, 0.5, 3, -1, 2, -0.5, 0]),
                "w": tensor((4, 3), [1, -1, 0.5, 2, 0, -0.5, -1, 1.5, 2, 0.25, -2, 1]),
                "b": tensor((3,), [-1, 0.5, 2]),
            }
        case "square-sum" | "grad-square-sum":
            return {"x": tensor((2, 3), [1, -2, 3, -4, 0.5, 2.5])}
        case "grad-dense-loss":
            return {
                "x": tensor((1, 2), [1.5, -2.0]),
                "w": tensor((2, 2), [0.25, -1.0, 2.0, 0.5]),
                "b": tensor((2,), [0.1, -0.2]),
            }
        case "grad-softmax-dense":
            return {
                "hidden": patterned_tensor((2, 8), 0.05, 2),
                "logits": patterned_tensor((2, 10), 0.2, 4),
                "labels": tensor(
                    (2, 10),
                    [
                        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                    ],
                ),
            }
        case "grad-relu-dense":
            return {
                "x": patterned_tensor((2, 784), 0.01, 1),
                "hidden_grad": patterned_tensor((2, 8), 0.03, 6),
                "relu_mask": tensor(
                    (2, 8),
                    [
                        1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0,
                        0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0,
                    ],
                ),
            }
        case "linear-train-step":
            return {
                "w": Tensor.scalar(1.0),
                "grad": Tensor.scalar(-4.0),
            }
        case "sgd-parameter-tree":
            return {
                "w": tensor((2, 2), [1.0, -2.0, 0.5, 3.0]),
                "b": tensor((2,), [0.25, -0.75]),
                "grad_w": tensor((2, 2), [0.5, -1.0, 2.0, -0.25]),
                "grad_b": tensor((2,), [1.5, -2.0]),
            }
        case "mnist-parameter-tree":
            return {
                "w1": patterned_tensor((784, 8), 0.002, 3),
                "b1": patterned_tensor((8,), 0.01, 5),
                "w2": patterned_tensor((8, 10), 0.02, 7),
                "b2": patterned_tensor((10,), 0.01, 9),
                "grad_w1": patterned_tensor((784, 8), 0.0005, 11),
                "grad_b1": patterned_tensor((8,), 0.001, 13),
                "grad_w2": patterned_tensor((8, 10), 0.002, 15),
                "grad_b2": patterned_tensor((10,), 0.001, 17),
            }
        case _:
            fail(f"unknown oracle case {name}")


def expected(name: str, inputs: dict[str, Tensor]) -> Tensor | list[Tensor]:
    match name:
        case "affine":
            shifted = elementwise(inputs["x"], inputs["bias"], lambda a, b: a + b)
            return elementwise(shifted, shifted, lambda a, b: a * b)
        case "matmul":
            return matmul(inputs["lhs"], inputs["rhs"])
        case "nn-primitives":
            shifted = elementwise(inputs["x"], broadcast_to(inputs["bias"], (2, 3)), lambda a, b: a + b)
            return Tensor.scalar(sum(value * 2.0 for value in shifted.data))
        case "mlp-forward":
            h = elementwise(matmul(inputs["x"], inputs["w1"]), broadcast_to(inputs["b1"], (2, 3)), lambda a, b: a + b)
            h2 = elementwise(h, h, lambda a, b: a * b)
            return elementwise(matmul(h2, inputs["w2"]), broadcast_to(inputs["b2"], (2, 2)), lambda a, b: a + b)
        case "relu-forward":
            h = elementwise(matmul(inputs["x"], inputs["w"]), broadcast_to(inputs["b"], (2, 3)), lambda a, b: a + b)
            return elementwise(h, broadcast_to(Tensor.scalar(0.0), (2, 3)), max)
        case "mnist-forward":
            hidden = elementwise(matmul(inputs["x"], inputs["w1"]), broadcast_to(inputs["b1"], (2, 8)), lambda a, b: a + b)
            activated = elementwise(hidden, broadcast_to(Tensor.scalar(0.0), (2, 8)), max)
            return elementwise(matmul(activated, inputs["w2"]), broadcast_to(inputs["b2"], (2, 10)), lambda a, b: a + b)
        case "cross-entropy-loss":
            exp_logits = Tensor(inputs["logits"].shape, tuple(math.exp(value) for value in inputs["logits"].data))
            denom = sum(exp_logits.data)
            probs = Tensor(exp_logits.shape, tuple(value / denom for value in exp_logits.data))
            log_probs = Tensor(probs.shape, tuple(math.log(value) for value in probs.data))
            weighted = elementwise(inputs["labels"], log_probs, lambda label, log_prob: label * log_prob)
            return Tensor.scalar(-sum(weighted.data))
        case "mnist-cross-entropy":
            exp_logits = Tensor(inputs["logits"].shape, tuple(math.exp(value) for value in inputs["logits"].data))
            denom = broadcast_to(reduce_last_dim(exp_logits), inputs["logits"].shape)
            probs = elementwise(exp_logits, denom, lambda value, row_total: value / row_total)
            log_probs = Tensor(probs.shape, tuple(math.log(value) for value in probs.data))
            weighted = elementwise(inputs["labels"], log_probs, lambda label, log_prob: label * log_prob)
            return Tensor.scalar(-sum(weighted.data) / inputs["logits"].shape[0])
        case "vmap-pointwise":
            summed = elementwise(inputs["x"], inputs["y"], lambda a, b: a + b)
            return elementwise(summed, summed, lambda a, b: a * b)
        case "vmap-dense":
            return elementwise(matmul(inputs["x"], inputs["w"]), broadcast_to(inputs["b"], (2, 3)), lambda a, b: a + b)
        case "square-sum":
            return Tensor.scalar(sum(value * value for value in inputs["x"].data))
        case "grad-square-sum":
            return Tensor(inputs["x"].shape, tuple(2.0 * value for value in inputs["x"].data))
        case "grad-dense-loss":
            residual = elementwise(matmul(inputs["x"], inputs["w"]), broadcast_to(inputs["b"], (1, 2)), lambda a, b: a + b)
            grad_out = Tensor(residual.shape, tuple(2.0 * value for value in residual.data))
            return matmul(transpose_2d(inputs["x"]), grad_out)
        case "grad-softmax-dense":
            exp_logits = Tensor(inputs["logits"].shape, tuple(math.exp(value) for value in inputs["logits"].data))
            denom = broadcast_to(reduce_last_dim(exp_logits), inputs["logits"].shape)
            probs = elementwise(exp_logits, denom, lambda value, row_total: value / row_total)
            delta = elementwise(probs, inputs["labels"], lambda prob, label: (prob - label) / inputs["logits"].shape[0])
            grad_w2 = matmul(transpose_2d(inputs["hidden"]), delta)
            grad_b2_keepdim = reduce_last_dim(transpose_2d(delta))
            grad_b2 = Tensor((10,), grad_b2_keepdim.data)
            return [grad_w2, grad_b2]
        case "grad-relu-dense":
            pre_activation_grad = elementwise(inputs["hidden_grad"], inputs["relu_mask"], lambda grad, mask: grad * mask)
            grad_w1 = matmul(transpose_2d(inputs["x"]), pre_activation_grad)
            grad_b1_keepdim = reduce_last_dim(transpose_2d(pre_activation_grad))
            grad_b1 = Tensor((8,), grad_b1_keepdim.data)
            return [grad_w1, grad_b1]
        case "linear-train-step":
            return Tensor.scalar(inputs["w"].data[0] + inputs["grad"].data[0] * -0.1)
        case "sgd-parameter-tree":
            next_w = elementwise(inputs["w"], inputs["grad_w"], lambda param, grad: param - 0.1 * grad)
            next_b = elementwise(inputs["b"], inputs["grad_b"], lambda param, grad: param - 0.1 * grad)
            return [next_w, next_b]
        case "mnist-parameter-tree":
            return [
                elementwise(inputs["w1"], inputs["grad_w1"], lambda param, grad: param - 0.1 * grad),
                elementwise(inputs["b1"], inputs["grad_b1"], lambda param, grad: param - 0.1 * grad),
                elementwise(inputs["w2"], inputs["grad_w2"], lambda param, grad: param - 0.1 * grad),
                elementwise(inputs["b2"], inputs["grad_b2"], lambda param, grad: param - 0.1 * grad),
            ]
        case _:
            fail(f"unknown expected case {name}")


def as_outputs(value: Tensor | list[Tensor]) -> list[Tensor]:
    if isinstance(value, Tensor):
        return [value]
    return value


def assert_close(actual: list[Tensor], want: list[Tensor]) -> None:
    if len(actual) != len(want):
        fail(f"result count {len(actual)} != expected {len(want)}")
    for output_index, (actual_tensor, want_tensor) in enumerate(zip(actual, want)):
        assert_tensor_close(actual_tensor, want_tensor, output_index)


def assert_tensor_close(actual: Tensor, want: Tensor, output_index: int) -> None:
    if actual.shape != want.shape:
        fail(f"result[{output_index}] shape {actual.shape} != expected {want.shape}")
    for index, (got, expected_value) in enumerate(zip(actual.data, want.data)):
        if not math.isclose(got, expected_value, rel_tol=1e-6, abs_tol=1e-6):
            fail(f"result[{output_index}][{index}] {got} != expected {expected_value}")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: numeric_oracles.py CASE GENERATED_MLIR", file=sys.stderr)
        return 2
    case = argv[1]
    text = Path(argv[2]).read_text(encoding="utf-8")
    inputs = oracle_inputs(case)
    assert_close(execute(text, inputs), as_outputs(expected(case, inputs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
