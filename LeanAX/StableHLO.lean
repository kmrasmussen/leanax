import LeanAX.IR

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

def affineModule : Module :=
  let shape := [2, 3]
  let x := sameShapeF32 "x" shape
  let bias := sameShapeF32 "bias" shape
  let sum := sameShapeF32 "sum" shape
  let out := sameShapeF32 "out" shape
  { name := "leanax_affine",
    functionName := "main",
    inputs := [x, bias],
    bindings := [
      { result := sum, kind := .add x bias },
      { result := out, kind := .multiply sum sum }
    ],
    returns := out }

def matmulModule : Module :=
  let lhs := tensor "lhs" .f32 [2, 4]
  let rhs := tensor "rhs" .f32 [4, 3]
  let out := tensor "out" .f32 [2, 3]
  { name := "leanax_matmul",
    functionName := "main",
    inputs := [lhs, rhs],
    bindings := [
      { result := out, kind := .dotGeneral lhs rhs }
    ],
    returns := out }

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

def moduleByName (name : String) : Option Module :=
  match name with
  | "affine" => some affineModule
  | "matmul" => some matmulModule
  | "bad-add-shape" => some badAddShapeModule
  | _ => none

def availableCases : List String :=
  ["affine", "matmul", "bad-add-shape"]

end LeanAX
