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
    if len(value.shape) > len(shape) or value.shape != shape[-len(value.shape) :]:
        fail(f"cannot broadcast {value.shape} to {shape}")

    out = []
    prefix = len(shape) - len(value.shape)
    for index in range(numel(shape)):
        coords = unflatten(index, shape)
        operand_coords = coords[prefix:]
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


def parse_signature(text: str) -> list[str]:
    signature = re.search(r"func\.func @main\((.*?)\) ->", text, re.S)
    if signature is None:
        fail("missing function signature")
    return re.findall(r"%([A-Za-z0-9_]+): tensor<[^>]+>", signature.group(1))


def parse_return(text: str) -> str:
    match = re.search(r"return %([A-Za-z0-9_]+) :", text)
    if match is None:
        fail("missing return")
    return match.group(1)


def result_shape(line: str) -> tuple[int, ...]:
    matches = re.findall(r"-> (tensor<[^>]+>)", line)
    if not matches:
        fail(f"missing result type in line: {line}")
    return parse_type(matches[-1])


def execute(text: str, inputs: dict[str, Tensor]) -> Tensor:
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
            values[name] = Tensor.scalar(sum(tensors[0].data))
        else:
            fail(f"unsupported op stablehlo.{op}")

        if values[name].shape != shape:
            fail(f"%{name} evaluated to {values[name].shape}, expected {shape}")

    return values[parse_return(text)]


def tensor(shape: tuple[int, ...], data: list[float]) -> Tensor:
    if len(data) != numel(shape):
        fail(f"bad oracle tensor for {shape}: {data}")
    return Tensor(shape, tuple(float(value) for value in data))


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
        case "vmap-pointwise":
            return {
                "x": tensor((4,), [1, 2, -1, 0.5]),
                "y": tensor((4,), [0.5, -3, 4, 1.5]),
            }
        case "square-sum" | "grad-square-sum":
            return {"x": tensor((2, 3), [1, -2, 3, -4, 0.5, 2.5])}
        case "linear-train-step":
            return {
                "w": Tensor.scalar(1.0),
                "grad": Tensor.scalar(-4.0),
            }
        case _:
            fail(f"unknown oracle case {name}")


def expected(name: str, inputs: dict[str, Tensor]) -> Tensor:
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
        case "vmap-pointwise":
            summed = elementwise(inputs["x"], inputs["y"], lambda a, b: a + b)
            return elementwise(summed, summed, lambda a, b: a * b)
        case "square-sum":
            return Tensor.scalar(sum(value * value for value in inputs["x"].data))
        case "grad-square-sum":
            return Tensor(inputs["x"].shape, tuple(2.0 * value for value in inputs["x"].data))
        case "linear-train-step":
            return Tensor.scalar(inputs["w"].data[0] + inputs["grad"].data[0] * -0.1)
        case _:
            fail(f"unknown expected case {name}")


def assert_close(actual: Tensor, want: Tensor) -> None:
    if actual.shape != want.shape:
        fail(f"result shape {actual.shape} != expected {want.shape}")
    for index, (got, expected_value) in enumerate(zip(actual.data, want.data)):
        if not math.isclose(got, expected_value, rel_tol=1e-6, abs_tol=1e-6):
            fail(f"result[{index}] {got} != expected {expected_value}")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: numeric_oracles.py CASE GENERATED_MLIR", file=sys.stderr)
        return 2
    case = argv[1]
    text = Path(argv[2]).read_text(encoding="utf-8")
    inputs = oracle_inputs(case)
    assert_close(execute(text, inputs), expected(case, inputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
