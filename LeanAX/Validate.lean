import LeanAX.IR

namespace LeanAX

inductive ValidationError where
  | duplicateInput (name : String)
  | duplicateResult (name : String)
  | undefinedReference (context : String) (name : String)
  | tensorMismatch (context : String) (lhs : TensorType) (rhs : TensorType)
  | dotInnerMismatch (lhsDim : Nat) (rhsDim : Nat)
  | dotResultShapeMismatch (expected : TensorType) (actual : TensorType)
  | dotUnsupportedTypeOrRank
  deriving Repr, BEq

def ValidationError.render : ValidationError -> String
  | .duplicateInput name =>
      s!"module inputs contain duplicate name %{name}"
  | .duplicateResult name =>
      s!"binding %{name}: duplicate result name"
  | .undefinedReference context name =>
      s!"{context}: undefined reference %{name}"
  | .tensorMismatch context lhs rhs =>
      s!"{context}: expected matching tensor types, got {lhs.stableName} and {rhs.stableName}"
  | .dotInnerMismatch lhsDim rhsDim =>
      s!"stablehlo.dot_general: inner dimensions differ ({lhsDim} vs {rhsDim})"
  | .dotResultShapeMismatch expected _actual =>
      s!"stablehlo.dot_general: result shape must be {expected.stableName}"
  | .dotUnsupportedTypeOrRank =>
      "stablehlo.dot_general: expected rank-2 f32 operands and rank-2 f32 result"

def namesOf (values : List ValueRef) : List String :=
  values.map (fun value => value.name)

def hasName (name : String) : List ValueRef -> Bool
  | [] => false
  | value :: rest => value.name == name || hasName name rest

def hasDuplicateString : List String -> Bool
  | [] => false
  | x :: xs => xs.contains x || hasDuplicateString xs

def firstDuplicateString : List String -> Option String
  | [] => none
  | x :: xs => if xs.contains x then some x else firstDuplicateString xs

def requireTensorEq (context : String) (lhs : TensorType) (rhs : TensorType) :
    Except ValidationError Unit :=
  if lhs == rhs then
    .ok ()
  else
    .error (.tensorMismatch context lhs rhs)

def requireDefined (context : String) (defined : List ValueRef) (value : ValueRef) :
    Except ValidationError Unit :=
  if hasName value.name defined then
    .ok ()
  else
    .error (.undefinedReference context value.name)

def validateBinding (defined : List ValueRef) (binding : Binding) :
    Except ValidationError Unit := do
  if hasName binding.result.name defined then
    throw (.duplicateResult binding.result.name)
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
            throw (.dotInnerMismatch kLeft kRight)
          if m != mOut || n != nOut then
            throw (.dotResultShapeMismatch (tensor "_expected" .f32 [m, n]).ty binding.result.ty)
      | _, _, _, _, _, _ =>
          throw .dotUnsupportedTypeOrRank

partial def validateBindings (defined : List ValueRef) : List Binding ->
    Except ValidationError (List ValueRef)
  | [] => .ok defined
  | binding :: rest => do
      validateBinding defined binding
      validateBindings (defined ++ [binding.result]) rest

def Module.validate (modu : Module) : Except ValidationError Unit := do
  if let some name := firstDuplicateString (namesOf modu.inputs) then
    throw (.duplicateInput name)
  let defined ← validateBindings modu.inputs modu.bindings
  requireDefined "module return" defined modu.returns

end LeanAX
