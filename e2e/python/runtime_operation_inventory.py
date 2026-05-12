from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


MANIFEST = Path("generated/mnist-train-step-derived-mask.mlir.manifest.json")
EXPECTED_INPUTS = [
    ("x", "tensor<2x784xf32>"),
    ("labels", "tensor<2x10xf32>"),
    ("w1", "tensor<784x8xf32>"),
    ("b1", "tensor<8xf32>"),
    ("w2", "tensor<8x10xf32>"),
    ("b2", "tensor<10xf32>"),
]
EXPECTED_OUTPUTS = [
    ("next_w1", "tensor<784x8xf32>"),
    ("next_b1", "tensor<8xf32>"),
    ("next_w2", "tensor<8x10xf32>"),
    ("next_b2", "tensor<10xf32>"),
    ("loss", "tensor<f32>"),
]
EXPECTED_OPS = {
    "stablehlo.add",
    "stablehlo.broadcast_in_dim",
    "stablehlo.compare",
    "stablehlo.constant",
    "stablehlo.divide",
    "stablehlo.dot_general",
    "stablehlo.exponential",
    "stablehlo.log",
    "stablehlo.multiply",
    "stablehlo.reduce",
    "stablehlo.reshape",
    "stablehlo.select",
    "stablehlo.transpose",
}

CURRENT_RUNTIME_FIXTURE_OPS = {
    "stablehlo.add",
    "stablehlo.compare",
    "stablehlo.constant",
    "stablehlo.divide",
    "stablehlo.dot_general",
    "stablehlo.exponential",
    "stablehlo.log",
    "stablehlo.multiply",
    "stablehlo.select",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def value_pairs(values: object, key: str) -> list[tuple[str, str]]:
    require(isinstance(values, list), f"{key} must be a list")
    pairs = []
    for index, value in enumerate(values):
        require(isinstance(value, dict), f"{key}[{index}] must be an object")
        name = value.get("name")
        typ = value.get("type")
        require(isinstance(name, str), f"{key}[{index}].name must be a string")
        require(isinstance(typ, str), f"{key}[{index}].type must be a string")
        pairs.append((name, typ))
    return pairs


def verify() -> None:
    require(MANIFEST.is_file(), f"missing lowering manifest {MANIFEST}")
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(data.get("schema") == "leanax.lowering.v1", "unexpected lowering manifest schema")
    require(data.get("module") == "leanax_mnist_train_step_derived_mask", "unexpected module")
    require(data.get("function") == "main", "unexpected function")
    require(value_pairs(data.get("inputs"), "inputs") == EXPECTED_INPUTS, "input contract drifted")
    require(value_pairs(data.get("outputs"), "outputs") == EXPECTED_OUTPUTS, "output contract drifted")

    operations = data.get("operations")
    require(isinstance(operations, list), "operations must be a list")
    op_names = []
    result_types = set()
    for index, operation in enumerate(operations):
        require(isinstance(operation, dict), f"operations[{index}] must be an object")
        op_name = operation.get("op")
        result_type = operation.get("result_type")
        require(isinstance(op_name, str), f"operations[{index}].op must be a string")
        require(isinstance(result_type, str), f"operations[{index}].result_type must be a string")
        op_names.append(op_name)
        result_types.add(result_type)

    actual_ops = set(op_names)
    require(actual_ops == EXPECTED_OPS, f"operation set drifted: {sorted(actual_ops)}")
    unsupported = sorted(actual_ops - CURRENT_RUNTIME_FIXTURE_OPS)
    counts = Counter(op_names)
    count_text = ",".join(f"{name}:{counts[name]}" for name in sorted(counts))
    unsupported_text = ",".join(unsupported)
    shape_text = ",".join(sorted(result_types))
    print(
        "runtime-operation-inventory "
        f"ops={len(actual_ops)} "
        f"bindings={len(op_names)} "
        f"unsupported={unsupported_text} "
        f"counts={count_text} "
        f"result_types={shape_text}"
    )


if __name__ == "__main__":
    verify()
