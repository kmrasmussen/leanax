import LeanAX.Build

namespace LeanAX

def TensorType.prependBatch (batch : Nat) (ty : TensorType) : TensorType :=
  { ty with shape := batch :: ty.shape }

def ValueRef.prependBatch (batch : Nat) (value : ValueRef) : ValueRef :=
  { value with ty := value.ty.prependBatch batch }

def Binding.prependPointwiseBatch (batch : Nat) (binding : Binding) :
    Except ValidationError Binding :=
  match binding.kind with
  | .add lhs rhs =>
      checkedAdd binding.result.name (lhs.prependBatch batch) (rhs.prependBatch batch)
  | .multiply lhs rhs =>
      checkedMultiply binding.result.name (lhs.prependBatch batch) (rhs.prependBatch batch)
  | _ =>
      throw (.unsupportedTransform "vmap" binding.result.name)

def Module.vmapPointwise (batch : Nat) (name : String) (modu : Module) :
    Except ValidationError Module := do
  let inputs := modu.inputs.map (ValueRef.prependBatch batch)
  let bindings ← modu.bindings.mapM (Binding.prependPointwiseBatch batch)
  let returns := modu.returns.prependBatch batch
  checkedModule name modu.functionName inputs bindings returns

def scalarPointwiseModule? : Except ValidationError Module := do
  let x := tensor "x" .f32 []
  let y := tensor "y" .f32 []
  let sum ← checkedAdd "sum" x y
  let out ← checkedMultiply "out" sum.result sum.result
  checkedModule "leanax_scalar_pointwise" "main" [x, y] [sum, out] out.result

def vmapPointwiseModule? : Except ValidationError Module := do
  let scalar ← scalarPointwiseModule?
  scalar.vmapPointwise 4 "leanax_vmap_pointwise"

end LeanAX
