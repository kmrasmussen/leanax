import LeanAX.IR

namespace LeanAX

def namesOf (values : List ValueRef) : List String :=
  values.map (fun value => value.name)

def hasName (name : String) : List ValueRef -> Bool
  | [] => false
  | value :: rest => value.name == name || hasName name rest

def hasDuplicateString : List String -> Bool
  | [] => false
  | x :: xs => xs.contains x || hasDuplicateString xs

def requireTensorEq (context : String) (lhs : TensorType) (rhs : TensorType) : Except String Unit :=
  if lhs == rhs then
    .ok ()
  else
    .error s!"{context}: expected matching tensor types, got {lhs.stableName} and {rhs.stableName}"

def requireDefined (context : String) (defined : List ValueRef) (value : ValueRef) : Except String Unit :=
  if hasName value.name defined then
    .ok ()
  else
    .error s!"{context}: undefined reference %{value.name}"

def validateBinding (defined : List ValueRef) (binding : Binding) : Except String Unit := do
  if hasName binding.result.name defined then
    throw s!"binding %{binding.result.name}: duplicate result name"
  match binding.kind with
  | .add lhs rhs =>
      requireDefined "stablehlo.add lhs" defined lhs
      requireDefined "stablehlo.add rhs" defined rhs
      requireTensorEq "stablehlo.add operands" lhs.ty rhs.ty
      requireTensorEq "stablehlo.add result" lhs.ty binding.result.ty
  | .multiply lhs rhs =>
      requireDefined "stablehlo.multiply lhs" defined lhs
      requireDefined "stablehlo.multiply rhs" defined rhs
      requireTensorEq "stablehlo.multiply operands" lhs.ty rhs.ty
      requireTensorEq "stablehlo.multiply result" lhs.ty binding.result.ty
  | .dotGeneral lhs rhs =>
      requireDefined "stablehlo.dot_general lhs" defined lhs
      requireDefined "stablehlo.dot_general rhs" defined rhs
      match lhs.ty.dtype, rhs.ty.dtype, binding.result.ty.dtype, lhs.ty.shape, rhs.ty.shape, binding.result.ty.shape with
      | .f32, .f32, .f32, [m, kLeft], [kRight, n], [mOut, nOut] =>
          if kLeft != kRight then
            throw s!"stablehlo.dot_general: inner dimensions differ ({kLeft} vs {kRight})"
          if m != mOut || n != nOut then
            throw s!"stablehlo.dot_general: result shape must be tensor<{m}x{n}xf32>"
      | _, _, _, _, _, _ =>
          throw "stablehlo.dot_general: expected rank-2 f32 operands and rank-2 f32 result"

partial def validateBindings (defined : List ValueRef) : List Binding -> Except String (List ValueRef)
  | [] => .ok defined
  | binding :: rest => do
      validateBinding defined binding
      validateBindings (defined ++ [binding.result]) rest

def Module.validate (modu : Module) : Except String Unit := do
  if hasDuplicateString (namesOf modu.inputs) then
    throw "module inputs contain duplicate names"
  let defined ← validateBindings modu.inputs modu.bindings
  requireDefined "module return" defined modu.returns

end LeanAX
