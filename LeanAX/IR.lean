namespace LeanAX

inductive DType where
  | f32
  | i32
  | pred
  deriving Repr, BEq

abbrev Shape := List Nat

structure TensorType where
  dtype : DType
  shape : Shape
  deriving Repr, BEq

structure ValueRef where
  name : String
  ty : TensorType
  deriving Repr, BEq

inductive BindingKind where
  | constant (value : String)
  | add (lhs : ValueRef) (rhs : ValueRef)
  | multiply (lhs : ValueRef) (rhs : ValueRef)
  | maximum (lhs : ValueRef) (rhs : ValueRef)
  | divide (lhs : ValueRef) (rhs : ValueRef)
  | exponential (operand : ValueRef)
  | logarithm (operand : ValueRef)
  | dotGeneral (lhs : ValueRef) (rhs : ValueRef)
  | broadcastInDim (operand : ValueRef)
  | reshape (operand : ValueRef)
  | transpose (operand : ValueRef) (permutation : List Nat)
  | reduceSum (operand : ValueRef)
  | reduceSumLastDim (operand : ValueRef)
  deriving Repr

structure Binding where
  result : ValueRef
  kind : BindingKind
  deriving Repr

structure Module where
  name : String
  functionName : String
  inputs : List ValueRef
  bindings : List Binding
  returns : List ValueRef
  deriving Repr

def DType.stableName : DType -> String
  | .f32 => "f32"
  | .i32 => "i32"
  | .pred => "i1"

def joinSep (sep : String) : List String -> String
  | [] => ""
  | [x] => x
  | x :: xs => x ++ sep ++ joinSep sep xs

def TensorType.stableName (ty : TensorType) : String :=
  match ty.shape with
  | [] => "tensor<" ++ ty.dtype.stableName ++ ">"
  | dims =>
      "tensor<" ++ joinSep "x" (dims.map (fun n => toString n)) ++ "x" ++
        ty.dtype.stableName ++ ">"

def ValueRef.percent (value : ValueRef) : String :=
  "%" ++ value.name

def ValueRef.parameter (value : ValueRef) : String :=
  value.percent ++ ": " ++ value.ty.stableName

def Shape.numElements : Shape -> Nat
  | [] => 1
  | dim :: rest => dim * Shape.numElements rest

def Shape.isSuffixOf (suffix : Shape) (full : Shape) : Bool :=
  if suffix == full then
    true
  else
    match full with
    | [] => false
    | _ :: rest => suffix.isSuffixOf rest

def Shape.broadcastableToSameRank : Shape -> Shape -> Bool
  | [], [] => true
  | sourceDim :: sourceRest, targetDim :: targetRest =>
      (sourceDim == targetDim || sourceDim == 1) && broadcastableToSameRank sourceRest targetRest
  | _, _ => false

def Shape.broadcastableTo (source : Shape) (target : Shape) : Bool :=
  source.isSuffixOf target || source.broadcastableToSameRank target

def tensor (name : String) (dtype : DType) (shape : Shape) : ValueRef :=
  { name := name, ty := { dtype := dtype, shape := shape } }

def sameShapeF32 (name : String) (shape : Shape) : ValueRef :=
  tensor name .f32 shape

end LeanAX
