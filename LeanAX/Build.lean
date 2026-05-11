import LeanAX.Validate

namespace LeanAX

def checkedAdd (resultName : String) (lhs : ValueRef) (rhs : ValueRef) :
    Except ValidationError Binding := do
  requireTensorEq "stablehlo.add operands" lhs.ty rhs.ty
  pure { result := tensor resultName lhs.ty.dtype lhs.ty.shape, kind := .add lhs rhs }

def checkedMultiply (resultName : String) (lhs : ValueRef) (rhs : ValueRef) :
    Except ValidationError Binding := do
  requireTensorEq "stablehlo.multiply operands" lhs.ty rhs.ty
  pure { result := tensor resultName lhs.ty.dtype lhs.ty.shape, kind := .multiply lhs rhs }

def checkedDotGeneral (resultName : String) (lhs : ValueRef) (rhs : ValueRef) :
    Except ValidationError Binding := do
  match lhs.ty.dtype, rhs.ty.dtype, lhs.ty.shape, rhs.ty.shape with
  | .f32, .f32, [m, kLeft], [kRight, n] =>
      if kLeft != kRight then
        throw (.dotInnerMismatch kLeft kRight)
      pure {
        result := tensor resultName .f32 [m, n],
        kind := .dotGeneral lhs rhs
      }
  | _, _, _, _ =>
      throw .dotUnsupportedTypeOrRank

def checkedModule
    (name : String)
    (functionName : String)
    (inputs : List ValueRef)
    (bindings : List Binding)
    (returns : ValueRef) :
    Except ValidationError Module := do
  let modu := {
    name := name,
    functionName := functionName,
    inputs := inputs,
    bindings := bindings,
    returns := returns
  }
  modu.validate
  pure modu

end LeanAX
