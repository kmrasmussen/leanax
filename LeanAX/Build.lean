import LeanAX.Validate

namespace LeanAX

def checkedConstant (resultName : String) (dtype : DType) (shape : Shape) (value : String) :
    Except ValidationError Binding := do
  pure { result := tensor resultName dtype shape, kind := .constant value }

def checkedAdd (resultName : String) (lhs : ValueRef) (rhs : ValueRef) :
    Except ValidationError Binding := do
  requireTensorEq "stablehlo.add operands" lhs.ty rhs.ty
  pure { result := tensor resultName lhs.ty.dtype lhs.ty.shape, kind := .add lhs rhs }

def checkedMultiply (resultName : String) (lhs : ValueRef) (rhs : ValueRef) :
    Except ValidationError Binding := do
  requireTensorEq "stablehlo.multiply operands" lhs.ty rhs.ty
  pure { result := tensor resultName lhs.ty.dtype lhs.ty.shape, kind := .multiply lhs rhs }

def checkedMaximum (resultName : String) (lhs : ValueRef) (rhs : ValueRef) :
    Except ValidationError Binding := do
  requireTensorEq "stablehlo.maximum operands" lhs.ty rhs.ty
  pure { result := tensor resultName lhs.ty.dtype lhs.ty.shape, kind := .maximum lhs rhs }

def checkedDivide (resultName : String) (lhs : ValueRef) (rhs : ValueRef) :
    Except ValidationError Binding := do
  requireTensorEq "stablehlo.divide operands" lhs.ty rhs.ty
  pure { result := tensor resultName lhs.ty.dtype lhs.ty.shape, kind := .divide lhs rhs }

def checkedExp (resultName : String) (operand : ValueRef) :
    Except ValidationError Binding := do
  pure { result := tensor resultName operand.ty.dtype operand.ty.shape, kind := .exponential operand }

def checkedLog (resultName : String) (operand : ValueRef) :
    Except ValidationError Binding := do
  pure { result := tensor resultName operand.ty.dtype operand.ty.shape, kind := .logarithm operand }

def checkedCompareGt (resultName : String) (lhs : ValueRef) (rhs : ValueRef) :
    Except ValidationError Binding := do
  requireTensorEq "stablehlo.compare operands" lhs.ty rhs.ty
  requireTensorEq "stablehlo.compare operand dtype" lhs.ty { dtype := .f32, shape := lhs.ty.shape }
  pure { result := tensor resultName .pred lhs.ty.shape, kind := .compareGt lhs rhs }

def checkedSelect (resultName : String) (predicate : ValueRef) (onTrue : ValueRef) (onFalse : ValueRef) :
    Except ValidationError Binding := do
  requireTensorEq "stablehlo.select values" onTrue.ty onFalse.ty
  requireTensorEq "stablehlo.select value dtype" onTrue.ty { dtype := .f32, shape := onTrue.ty.shape }
  requireTensorEq "stablehlo.select predicate" predicate.ty { dtype := .pred, shape := onTrue.ty.shape }
  pure { result := tensor resultName .f32 onTrue.ty.shape, kind := .select predicate onTrue onFalse }

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

def checkedBroadcastInDim (resultName : String) (operand : ValueRef) (shape : Shape) :
    Except ValidationError Binding := do
  let result := tensor resultName operand.ty.dtype shape
  requireBroadcastable operand.ty result.ty
  pure { result := result, kind := .broadcastInDim operand }

def checkedReshape (resultName : String) (operand : ValueRef) (shape : Shape) :
    Except ValidationError Binding := do
  let result := tensor resultName operand.ty.dtype shape
  requireSameElementCount operand.ty result.ty
  pure { result := result, kind := .reshape operand }

def checkedTranspose (resultName : String) (operand : ValueRef) (permutation : List Nat) :
    Except ValidationError Binding := do
  let shape ←
    match operand.ty.shape, permutation with
    | [m, n], [1, 0] => pure [n, m]
    | _, _ => throw (.transposeUnsupported operand.ty operand.ty permutation)
  let result := tensor resultName operand.ty.dtype shape
  requireTranspose operand.ty result.ty permutation
  pure { result := result, kind := .transpose operand permutation }

def checkedReduceSum (resultName : String) (operand : ValueRef) :
    Except ValidationError Binding := do
  let result := tensor resultName operand.ty.dtype []
  requireReduceSum operand.ty result.ty
  pure { result := result, kind := .reduceSum operand }

def checkedReduceSumLastDim (resultName : String) (operand : ValueRef) :
    Except ValidationError Binding := do
  let shape ←
    match operand.ty.shape with
    | [rows, _cols] => pure [rows, 1]
    | _ => throw (.reduceSumLastDimResultMismatch operand.ty (tensor resultName operand.ty.dtype []).ty)
  let result := tensor resultName operand.ty.dtype shape
  requireReduceSumLastDim operand.ty result.ty
  pure { result := result, kind := .reduceSumLastDim operand }

def checkedModuleMulti
    (name : String)
    (functionName : String)
    (inputs : List ValueRef)
    (bindings : List Binding)
    (returns : List ValueRef) :
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

def checkedModule
    (name : String)
    (functionName : String)
    (inputs : List ValueRef)
    (bindings : List Binding)
    (returns : ValueRef) :
    Except ValidationError Module :=
  checkedModuleMulti name functionName inputs bindings [returns]

end LeanAX
