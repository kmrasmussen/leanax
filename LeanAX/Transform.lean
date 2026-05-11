import LeanAX.Build
import LeanAX.DSL

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
  | .maximum lhs rhs =>
      checkedMaximum binding.result.name (lhs.prependBatch batch) (rhs.prependBatch batch)
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

def vmapDenseLayer
    (batch : Nat)
    (layerPrefix : String)
    (exampleInput : ValueRef)
    (weight : ValueRef)
    (bias : ValueRef) :
    Except ValidationError DSL.LayerResult := do
  match exampleInput.ty.shape with
  | [features] =>
      let batchedInput := { exampleInput with ty := { exampleInput.ty with shape := [batch, features] } }
      DSL.denseLayer layerPrefix batchedInput weight bias
  | _ =>
      throw (.unsupportedTransform "vmap-dense" exampleInput.name)

def vmapDenseModule? : Except ValidationError Module := do
  let x := tensor "x" .f32 [4]
  let w := tensor "w" .f32 [4, 3]
  let b := tensor "b" .f32 [3]
  let batchedX := x.prependBatch 2
  let dense ← vmapDenseLayer 2 "dense" x w b
  checkedModule "leanax_vmap_dense" "main" [batchedX, w, b] dense.bindings dense.output

end LeanAX
