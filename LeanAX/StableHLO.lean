import LeanAX.Build

namespace LeanAX

def BindingKind.render (kind : BindingKind) : String :=
  match kind with
  | .add lhs rhs => "stablehlo.add " ++ lhs.percent ++ ", " ++ rhs.percent
  | .multiply lhs rhs => "stablehlo.multiply " ++ lhs.percent ++ ", " ++ rhs.percent
  | .dotGeneral lhs rhs =>
      "stablehlo.dot_general " ++ lhs.percent ++ ", " ++ rhs.percent ++
        ", batching_dims = [] x []"

def Binding.render (binding : Binding) : String :=
  "    " ++ binding.result.percent ++ " = " ++ binding.kind.render ++
    " : " ++ binding.result.ty.stableName

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

def moduleByName (name : String) : Option Module :=
  match name with
  | "affine" => affineModule?.toOption
  | "matmul" => matmulModule?.toOption
  | "bad-add-shape" => some badAddShapeModule
  | "duplicate-input" => some duplicateInputModule
  | "undefined-ref" => some undefinedReferenceModule
  | "duplicate-result" => some duplicateResultModule
  | "bad-dot-inner" => some badDotInnerModule
  | "bad-dot-result" => some badDotResultModule
  | "bad-dot-rank" => some badDotRankModule
  | _ => none

def availableCases : List String :=
  [
    "affine",
    "matmul",
    "bad-add-shape",
    "duplicate-input",
    "undefined-ref",
    "duplicate-result",
    "bad-dot-inner",
    "bad-dot-result",
    "bad-dot-rank"
  ]

end LeanAX
