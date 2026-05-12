from __future__ import annotations

import json
from pathlib import Path


MANIFEST = Path("generated/mnist-train-step-derived-mask.mlir.manifest.json")
OUTPUT = Path("generated/runtime-scaling-budget.json")

SCHEMA = "leanax.runtime_scaling_budget.v1"
MAX_DEFAULT_LINES = 250_000
MAX_DEFAULT_SCALARS = 80_000
F32_TOLERANCE = 1.0e-4

EXPECTED_INPUTS = {
    "x": "tensor<2x784xf32>",
    "labels": "tensor<2x10xf32>",
    "w1": "tensor<784x8xf32>",
    "b1": "tensor<8xf32>",
    "w2": "tensor<8x10xf32>",
    "b2": "tensor<10xf32>",
}
EXPECTED_OUTPUTS = {
    "next_w1": "tensor<784x8xf32>",
    "next_b1": "tensor<8xf32>",
    "next_w2": "tensor<8x10xf32>",
    "next_b2": "tensor<10xf32>",
    "loss": "tensor<f32>",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_manifest() -> dict[str, object]:
    require(MANIFEST.is_file(), f"missing lowering manifest {MANIFEST}")
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(data.get("schema") == "leanax.lowering.v1", "unexpected lowering manifest schema")
    require(data.get("module") == "leanax_mnist_train_step_derived_mask", "unexpected module")
    return data


def value_map(values: object, key: str) -> dict[str, str]:
    require(isinstance(values, list), f"{key} must be a list")
    result: dict[str, str] = {}
    for index, value in enumerate(values):
        require(isinstance(value, dict), f"{key}[{index}] must be an object")
        name = value.get("name")
        typ = value.get("type")
        require(isinstance(name, str), f"{key}[{index}].name must be a string")
        require(isinstance(typ, str), f"{key}[{index}].type must be a string")
        result[name] = typ
    return result


def estimate() -> dict[str, object]:
    batch = 2
    inputs = 784
    hidden = 8
    classes = 10

    dense1_mul = batch * hidden * inputs
    dense1_add = batch * hidden * (inputs - 1) + batch * hidden
    relu = batch * hidden * 2
    dense2_mul = batch * classes * hidden
    dense2_add = batch * classes * (hidden - 1) + batch * classes

    forward_scalars = dense1_mul + dense1_add + relu + dense2_mul + dense2_add

    loss_scalars = (
        forward_scalars
        + batch * classes  # exp
        + batch * (classes - 1)  # denominator reductions
        + batch * classes  # probability divides
        + batch * classes  # label-weighted log probabilities
        + batch * (classes - 1)  # class reductions
        + (batch - 1)  # mean reduction
        + 1  # divide by batch
    )

    grad_w2 = hidden * classes * (batch * 2)
    grad_b2 = classes * (batch - 1)
    hidden_grad = batch * hidden * classes * 2
    pre_grad = batch * hidden
    grad_w1 = inputs * hidden * (batch * 2)
    grad_b1 = hidden * (batch - 1)
    gradient_scalars = loss_scalars + grad_w2 + grad_b2 + hidden_grad + pre_grad + grad_w1 + grad_b1

    updates = inputs * hidden * 2 + hidden * 2 + hidden * classes * 2 + classes * 2
    checksum = inputs * hidden + hidden + hidden * classes + classes + 5
    train_step_scalars = gradient_scalars + updates + checksum

    def case(name: str, scalar_ops: int) -> dict[str, object]:
        estimated_lines = scalar_ops + 96
        return {
            "name": name,
            "estimated_scalar_ops": scalar_ops,
            "estimated_mlir_lines": estimated_lines,
            "default_gate_candidate": scalar_ops <= MAX_DEFAULT_SCALARS and estimated_lines <= MAX_DEFAULT_LINES,
        }

    cases = [
        case("exact_forward", forward_scalars),
        case("exact_loss", loss_scalars),
        case("exact_gradient", gradient_scalars),
        case("exact_train_step", train_step_scalars),
    ]
    return {
        "schema": SCHEMA,
        "classifier": {
            "batch": batch,
            "inputs": inputs,
            "hidden": hidden,
            "classes": classes,
        },
        "limits": {
            "max_default_scalar_ops": MAX_DEFAULT_SCALARS,
            "max_default_mlir_lines": MAX_DEFAULT_LINES,
            "f32_tolerance": F32_TOLERANCE,
        },
        "cases": cases,
        "decision": {
            "route": "scalarized-llvm-mlir-runner",
            "default_gate_policy": "exact-shape runtime cases may enter the default gate while measured generated lines and runner time stay under the recorded limits",
            "fallback": "if measured runtime exceeds the budget, keep the case opt-in and add loop/buffer lowering before flipping direct_mnist_external_runtime",
        },
    }


def verify() -> None:
    manifest = load_manifest()
    require(value_map(manifest.get("inputs"), "inputs") == EXPECTED_INPUTS, "input contract drifted")
    require(value_map(manifest.get("outputs"), "outputs") == EXPECTED_OUTPUTS, "output contract drifted")
    report = estimate()
    cases = report["cases"]
    require(isinstance(cases, list), "cases must be a list")
    exact_train_step = next(case for case in cases if case["name"] == "exact_train_step")
    require(bool(exact_train_step["default_gate_candidate"]), "exact train-step estimate exceeds default-gate budget")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "runtime-scaling-budget "
        f"route={report['decision']['route']} "
        f"cases={len(cases)} "
        f"exact_train_step_scalar_ops={exact_train_step['estimated_scalar_ops']} "
        f"exact_train_step_mlir_lines={exact_train_step['estimated_mlir_lines']} "
        f"output={OUTPUT}"
    )


if __name__ == "__main__":
    verify()
