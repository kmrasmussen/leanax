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
  | add (lhs : ValueRef) (rhs : ValueRef)
  | multiply (lhs : ValueRef) (rhs : ValueRef)
  | dotGeneral (lhs : ValueRef) (rhs : ValueRef)
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
  returns : ValueRef
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

def tensor (name : String) (dtype : DType) (shape : Shape) : ValueRef :=
  { name := name, ty := { dtype := dtype, shape := shape } }

def sameShapeF32 (name : String) (shape : Shape) : ValueRef :=
  tensor name .f32 shape

end LeanAX
