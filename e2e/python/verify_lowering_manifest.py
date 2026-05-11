from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


def fail(message: str) -> None:
    print(f"lowering manifest verification failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    require(isinstance(data, dict), "manifest root must be an object")
    return data


def value_names(values: Any, field: str) -> list[str]:
    require(isinstance(values, list), f"{field} must be a list")
    names = []
    for index, value in enumerate(values):
        require(isinstance(value, dict), f"{field}[{index}] must be an object")
        name = value.get("name")
        ty = value.get("type")
        require(isinstance(name, str) and name, f"{field}[{index}].name must be a string")
        require(isinstance(ty, str) and ty.startswith("tensor<"), f"{field}[{index}].type must be a tensor type")
        names.append(name)
    return names


def mlir_module_name(text: str) -> str:
    match = re.search(r"^module @([A-Za-z0-9_]+) \{$", text, re.M)
    require(match is not None, "generated MLIR is missing module header")
    return match.group(1)


def mlir_inputs(text: str) -> list[str]:
    signature = re.search(r"func\.func @main\((.*?)\) ->", text, re.S)
    require(signature is not None, "generated MLIR is missing main signature")
    return re.findall(r"%([A-Za-z0-9_]+): tensor<[^>]+>", signature.group(1))


def mlir_return(text: str) -> str:
    match = re.search(r"return %([A-Za-z0-9_]+) :", text)
    require(match is not None, "generated MLIR is missing return")
    return match.group(1)


def mlir_operations(text: str) -> list[dict[str, Any]]:
    operations = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = re.search(r'%([A-Za-z0-9_]+) = "stablehlo\.([A-Za-z0-9_]+)"\(([^)]*)\).* -> (tensor<[^>]+>)', line)
        if match is None:
            continue
        result, op_name, operand_text, result_type = match.groups()
        operations.append(
            {
                "result": result,
                "op": f"stablehlo.{op_name}",
                "operands": re.findall(r"%([A-Za-z0-9_]+)", operand_text),
                "result_type": result_type,
                "mlir_line": line_number,
            }
        )
    require(operations, "generated MLIR has no stablehlo operations")
    return operations


def verify(manifest_path: Path, generated_path: Path) -> None:
    manifest = load_object(manifest_path)
    text = generated_path.read_text(encoding="utf-8")

    require(manifest.get("schema") == "leanax.lowering.v1", "unexpected schema")
    require(manifest.get("generated") == str(generated_path), "generated path mismatch")
    require(manifest.get("module") == mlir_module_name(text), "module name mismatch")
    require(manifest.get("function") == "main", "function name mismatch")
    require(value_names(manifest.get("inputs"), "inputs") == mlir_inputs(text), "input names mismatch")
    require(value_names(manifest.get("outputs"), "outputs") == [mlir_return(text)], "output names mismatch")

    manifest_ops = manifest.get("operations")
    require(isinstance(manifest_ops, list), "operations must be a list")
    generated_ops = mlir_operations(text)
    require(len(manifest_ops) == len(generated_ops), "operation count mismatch")

    for index, (manifest_op, generated_op) in enumerate(zip(manifest_ops, generated_ops)):
        require(isinstance(manifest_op, dict), f"operations[{index}] must be an object")
        require(manifest_op.get("id") == f"op{index}", f"operations[{index}].id mismatch")
        for field in ["result", "op", "operands", "result_type", "mlir_line"]:
            require(manifest_op.get(field) == generated_op[field], f"operations[{index}].{field} mismatch")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: verify_lowering_manifest.py MANIFEST GENERATED_MLIR", file=sys.stderr)
        return 2
    verify(Path(argv[1]), Path(argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
