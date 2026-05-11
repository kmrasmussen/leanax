import LeanAX.Build
import LeanAX.DSL
import LeanAX.Grad
import LeanAX.Training
import LeanAX.Transform

namespace LeanAX

def renderOperandTypes (values : List ValueRef) : String :=
  joinSep ", " (values.map (fun value => value.ty.stableName))

def renderOperands (values : List ValueRef) : String :=
  joinSep ", " (values.map ValueRef.percent)

def renderPermutation (permutation : List Nat) : String :=
  "[" ++ joinSep ", " (permutation.map (fun dim => toString dim)) ++ "]"

def renderGenericOp
    (result : ValueRef)
    (opName : String)
    (operands : List ValueRef)
    (attrs : Option String := none) : String :=
  let attrText :=
    match attrs with
    | none => ""
    | some text => " " ++ text
  "    " ++ result.percent ++ " = \"stablehlo." ++ opName ++ "\"(" ++
    renderOperands operands ++ ")" ++ attrText ++ " : (" ++
    renderOperandTypes operands ++ ") -> " ++ result.ty.stableName

def BindingKind.stableOpName : BindingKind -> String
  | .constant _ => "stablehlo.constant"
  | .add _ _ => "stablehlo.add"
  | .multiply _ _ => "stablehlo.multiply"
  | .maximum _ _ => "stablehlo.maximum"
  | .dotGeneral _ _ => "stablehlo.dot_general"
  | .broadcastInDim _ => "stablehlo.broadcast_in_dim"
  | .reshape _ => "stablehlo.reshape"
  | .transpose _ _ => "stablehlo.transpose"
  | .reduceSum _ => "stablehlo.reduce"

def BindingKind.operands : BindingKind -> List ValueRef
  | .constant _ => []
  | .add lhs rhs => [lhs, rhs]
  | .multiply lhs rhs => [lhs, rhs]
  | .maximum lhs rhs => [lhs, rhs]
  | .dotGeneral lhs rhs => [lhs, rhs]
  | .broadcastInDim operand => [operand]
  | .reshape operand => [operand]
  | .transpose operand _ => [operand]
  | .reduceSum operand => [operand]

def Binding.render (binding : Binding) : String :=
  match binding.kind with
  | .constant value =>
      "    " ++ binding.result.percent ++
        " = \"stablehlo.constant\"() {value = \"" ++ value ++
        "\"} : () -> " ++ binding.result.ty.stableName
  | .add lhs rhs =>
      renderGenericOp binding.result "add" [lhs, rhs]
  | .multiply lhs rhs =>
      renderGenericOp binding.result "multiply" [lhs, rhs]
  | .maximum lhs rhs =>
      renderGenericOp binding.result "maximum" [lhs, rhs]
  | .dotGeneral lhs rhs =>
      renderGenericOp binding.result "dot_general" [lhs, rhs] (some "{batching_dims = \"[] x []\"}")
  | .broadcastInDim operand =>
      renderGenericOp binding.result "broadcast_in_dim" [operand]
  | .reshape operand =>
      renderGenericOp binding.result "reshape" [operand]
  | .transpose operand permutation =>
      renderGenericOp binding.result "transpose" [operand]
        (some ("{permutation = " ++ renderPermutation permutation ++ "}"))
  | .reduceSum operand =>
      renderGenericOp binding.result "reduce" [operand] (some "{dimensions = \"all\"}")

def Module.renderSignature (modu : Module) : String :=
  "  func.func @" ++ modu.functionName ++ "(" ++
    joinSep ", " (modu.inputs.map ValueRef.parameter) ++ ") -> " ++
    modu.returns.ty.stableName ++ " {"

def Module.render (modu : Module) : String :=
  let lines :=
    ["module @" ++ modu.name ++ " {",
     modu.renderSignature] ++
    (modu.bindings.map Binding.render) ++
    ["    return " ++ modu.returns.percent ++ " : " ++ modu.returns.ty.stableName,
     "  }",
     "}"]
  joinSep "\n" lines ++ "\n"

def jsonEscapeChar : Char -> String
  | '"' => "\\\""
  | '\\' => "\\\\"
  | '\n' => "\\n"
  | '\r' => "\\r"
  | '\t' => "\\t"
  | c => String.singleton c

def jsonString (value : String) : String :=
  "\"" ++ joinSep "" (value.toList.map jsonEscapeChar) ++ "\""

def jsonArray (items : List String) : String :=
  "[" ++ joinSep ", " items ++ "]"

def ValueRef.renderManifest (value : ValueRef) : String :=
  "{\"name\": " ++ jsonString value.name ++
    ", \"type\": " ++ jsonString value.ty.stableName ++ "}"

def Binding.renderManifest (index : Nat) (binding : Binding) : String :=
  let operands := binding.kind.operands.map (fun value => jsonString value.name)
  "    {\n" ++
    "      \"id\": " ++ jsonString s!"op{index}" ++ ",\n" ++
    "      \"result\": " ++ jsonString binding.result.name ++ ",\n" ++
    "      \"op\": " ++ jsonString binding.kind.stableOpName ++ ",\n" ++
    "      \"operands\": " ++ jsonArray operands ++ ",\n" ++
    "      \"result_type\": " ++ jsonString binding.result.ty.stableName ++ ",\n" ++
    "      \"mlir_line\": " ++ toString (index + 3) ++ "\n" ++
    "    }"

def enumerateBindings : Nat -> List Binding -> List String
  | _, [] => []
  | index, binding :: rest => binding.renderManifest index :: enumerateBindings (index + 1) rest

def Module.renderLoweringManifest (modu : Module) (generatedPath : String) : String :=
  "{\n" ++
    "  \"schema\": \"leanax.lowering.v1\",\n" ++
    "  \"generated\": " ++ jsonString generatedPath ++ ",\n" ++
    "  \"module\": " ++ jsonString modu.name ++ ",\n" ++
    "  \"function\": " ++ jsonString modu.functionName ++ ",\n" ++
    "  \"inputs\": " ++ jsonArray (modu.inputs.map ValueRef.renderManifest) ++ ",\n" ++
    "  \"outputs\": " ++ jsonArray [modu.returns.renderManifest] ++ ",\n" ++
    "  \"operations\": [\n" ++
    joinSep ",\n" (enumerateBindings 0 modu.bindings) ++ "\n" ++
    "  ]\n" ++
    "}\n"

def affineModule? : Except ValidationError Module := do
  let shape := [2, 3]
  let x := sameShapeF32 "x" shape
  let bias := sameShapeF32 "bias" shape
  let sum ← checkedAdd "sum" x bias
  let out ← checkedMultiply "out" sum.result sum.result
  checkedModule "leanax_affine" "main" [x, bias] [sum, out] out.result

def matmulModule? : Except ValidationError Module := do
  let lhs := tensor "lhs" .f32 [2, 4]
  let rhs := tensor "rhs" .f32 [4, 3]
  let out ← checkedDotGeneral "out" lhs rhs
  checkedModule "leanax_matmul" "main" [lhs, rhs] [out] out.result

def nnPrimitivesModule? : Except ValidationError Module := do
  let x := tensor "x" .f32 [2, 3]
  let bias := tensor "bias" .f32 [3]
  let scale ← checkedConstant "scale" .f32 [] "2.0"
  let biasBatched ← checkedBroadcastInDim "bias_batched" bias [2, 3]
  let shifted ← checkedAdd "shifted" x biasBatched.result
  let flat ← checkedReshape "flat" shifted.result [6]
  let restored ← checkedReshape "restored" flat.result [2, 3]
  let transposed ← checkedTranspose "transposed" restored.result [1, 0]
  let scaleBatched ← checkedBroadcastInDim "scale_batched" scale.result [2, 3]
  let scaled ← checkedMultiply "scaled" shifted.result scaleBatched.result
  let loss ← checkedReduceSum "loss" scaled.result
  checkedModule "leanax_nn_primitives" "main" [x, bias] [
    scale,
    biasBatched,
    shifted,
    flat,
    restored,
    transposed,
    scaleBatched,
    scaled,
    loss
  ] loss.result

def badAddShapeModule : Module :=
  let x := tensor "x" .f32 [2, 3]
  let y := tensor "y" .f32 [3, 2]
  let out := tensor "out" .f32 [2, 3]
  { name := "leanax_bad_add_shape",
    functionName := "main",
    inputs := [x, y],
    bindings := [
      { result := out, kind := .add x y }
    ],
    returns := out }

def duplicateInputModule : Module :=
  let x := tensor "x" .f32 [2, 3]
  { name := "leanax_duplicate_input",
    functionName := "main",
    inputs := [x, x],
    bindings := [],
    returns := x }

def undefinedReferenceModule : Module :=
  let x := tensor "x" .f32 [2, 3]
  let y := tensor "y" .f32 [2, 3]
  let out := tensor "out" .f32 [2, 3]
  { name := "leanax_undefined_ref",
    functionName := "main",
    inputs := [y],
    bindings := [{ result := out, kind := .add x y }],
    returns := out }

def duplicateResultModule : Module :=
  let x := tensor "x" .f32 [2, 3]
  let y := tensor "y" .f32 [2, 3]
  let sum := tensor "sum" .f32 [2, 3]
  let out := tensor "out" .f32 [2, 3]
  { name := "leanax_duplicate_result",
    functionName := "main",
    inputs := [x, y],
    bindings := [
      { result := sum, kind := .add x y },
      { result := sum, kind := .multiply x y },
      { result := out, kind := .add sum y }
    ],
    returns := out }

def badDotInnerModule : Module :=
  let lhs := tensor "lhs" .f32 [2, 4]
  let rhs := tensor "rhs" .f32 [5, 3]
  let out := tensor "out" .f32 [2, 3]
  { name := "leanax_bad_dot_inner",
    functionName := "main",
    inputs := [lhs, rhs],
    bindings := [{ result := out, kind := .dotGeneral lhs rhs }],
    returns := out }

def badDotResultModule : Module :=
  let lhs := tensor "lhs" .f32 [2, 4]
  let rhs := tensor "rhs" .f32 [4, 3]
  let out := tensor "out" .f32 [2, 4]
  { name := "leanax_bad_dot_result",
    functionName := "main",
    inputs := [lhs, rhs],
    bindings := [{ result := out, kind := .dotGeneral lhs rhs }],
    returns := out }

def badDotRankModule : Module :=
  let lhs := tensor "lhs" .f32 [4]
  let rhs := tensor "rhs" .f32 [4, 3]
  let out := tensor "out" .f32 [3]
  { name := "leanax_bad_dot_rank",
    functionName := "main",
    inputs := [lhs, rhs],
    bindings := [{ result := out, kind := .dotGeneral lhs rhs }],
    returns := out }

def badBroadcastModule : Module :=
  let bias := tensor "bias" .f32 [2]
  let out := tensor "out" .f32 [2, 3]
  { name := "leanax_bad_broadcast",
    functionName := "main",
    inputs := [bias],
    bindings := [{ result := out, kind := .broadcastInDim bias }],
    returns := out }

def badReshapeModule : Module :=
  let x := tensor "x" .f32 [2, 3]
  let out := tensor "out" .f32 [5]
  { name := "leanax_bad_reshape",
    functionName := "main",
    inputs := [x],
    bindings := [{ result := out, kind := .reshape x }],
    returns := out }

def badTransposeModule : Module :=
  let x := tensor "x" .f32 [2, 3]
  let out := tensor "out" .f32 [2, 3]
  { name := "leanax_bad_transpose",
    functionName := "main",
    inputs := [x],
    bindings := [{ result := out, kind := .transpose x [1, 0] }],
    returns := out }

def badReduceModule : Module :=
  let x := tensor "x" .f32 [2, 3]
  let out := tensor "out" .f32 [2]
  { name := "leanax_bad_reduce",
    functionName := "main",
    inputs := [x],
    bindings := [{ result := out, kind := .reduceSum x }],
    returns := out }

def badMaximumShapeModule : Module :=
  let x := tensor "x" .f32 [2, 3]
  let y := tensor "y" .f32 [3, 2]
  let out := tensor "out" .f32 [2, 3]
  { name := "leanax_bad_maximum_shape",
    functionName := "main",
    inputs := [x, y],
    bindings := [{ result := out, kind := .maximum x y }],
    returns := out }

def moduleByName (name : String) : Option Module :=
  match name with
  | "affine" => affineModule?.toOption
  | "matmul" => matmulModule?.toOption
  | "nn-primitives" => nnPrimitivesModule?.toOption
  | "mlp-forward" => DSL.mlpForwardModule?.toOption
  | "relu-forward" => DSL.reluForwardModule?.toOption
  | "vmap-pointwise" => vmapPointwiseModule?.toOption
  | "square-sum" => squareSumModule?.toOption
  | "grad-square-sum" => gradSquareSumModule?.toOption
  | "linear-train-step" => linearTrainStepModule?.toOption
  | "bad-add-shape" => some badAddShapeModule
  | "duplicate-input" => some duplicateInputModule
  | "undefined-ref" => some undefinedReferenceModule
  | "duplicate-result" => some duplicateResultModule
  | "bad-dot-inner" => some badDotInnerModule
  | "bad-dot-result" => some badDotResultModule
  | "bad-dot-rank" => some badDotRankModule
  | "bad-broadcast" => some badBroadcastModule
  | "bad-reshape" => some badReshapeModule
  | "bad-transpose" => some badTransposeModule
  | "bad-reduce" => some badReduceModule
  | "bad-maximum-shape" => some badMaximumShapeModule
  | _ => none

def availableCases : List String :=
  [
    "affine",
    "matmul",
    "nn-primitives",
    "mlp-forward",
    "relu-forward",
    "vmap-pointwise",
    "square-sum",
    "grad-square-sum",
    "linear-train-step",
    "bad-add-shape",
    "duplicate-input",
    "undefined-ref",
    "duplicate-result",
    "bad-dot-inner",
    "bad-dot-result",
    "bad-dot-rank",
    "bad-broadcast",
    "bad-reshape",
    "bad-transpose",
    "bad-reduce",
    "bad-maximum-shape"
  ]

end LeanAX
