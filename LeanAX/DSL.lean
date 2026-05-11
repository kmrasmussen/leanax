import LeanAX.Build

namespace LeanAX
namespace DSL

structure LayerResult where
  bindings : List Binding
  output : ValueRef

def denseLayer
    (layerPrefix : String)
    (input : ValueRef)
    (weight : ValueRef)
    (bias : ValueRef) :
    Except ValidationError LayerResult := do
  let linear ← checkedDotGeneral (layerPrefix ++ "_linear") input weight
  let biasBatched ← checkedBroadcastInDim (layerPrefix ++ "_bias") bias linear.result.ty.shape
  let shifted ← checkedAdd (layerPrefix ++ "_out") linear.result biasBatched.result
  pure { bindings := [linear, biasBatched, shifted], output := shifted.result }

def squareActivation (layerPrefix : String) (input : ValueRef) :
    Except ValidationError LayerResult := do
  let out ← checkedMultiply (layerPrefix ++ "_out") input input
  pure { bindings := [out], output := out.result }

def reluActivation (layerPrefix : String) (input : ValueRef) :
    Except ValidationError LayerResult := do
  let zero ← checkedConstant (layerPrefix ++ "_zero") input.ty.dtype [] "0.0"
  let zeroBatched ← checkedBroadcastInDim (layerPrefix ++ "_zero_batched") zero.result input.ty.shape
  let out ← checkedMaximum (layerPrefix ++ "_out") input zeroBatched.result
  pure { bindings := [zero, zeroBatched, out], output := out.result }

def mlpForwardModule? : Except ValidationError Module := do
  let x := tensor "x" .f32 [2, 4]
  let w1 := tensor "w1" .f32 [4, 3]
  let b1 := tensor "b1" .f32 [3]
  let w2 := tensor "w2" .f32 [3, 2]
  let b2 := tensor "b2" .f32 [2]
  let hidden ← denseLayer "hidden" x w1 b1
  let activated ← squareActivation "activation" hidden.output
  let logits ← denseLayer "logits" activated.output w2 b2
  checkedModule "leanax_mlp_forward" "main" [x, w1, b1, w2, b2]
    (hidden.bindings ++ activated.bindings ++ logits.bindings)
    logits.output

def reluForwardModule? : Except ValidationError Module := do
  let x := tensor "x" .f32 [2, 4]
  let w := tensor "w" .f32 [4, 3]
  let b := tensor "b" .f32 [3]
  let hidden ← denseLayer "hidden" x w b
  let activated ← reluActivation "relu" hidden.output
  checkedModule "leanax_relu_forward" "main" [x, w, b]
    (hidden.bindings ++ activated.bindings)
    activated.output

def mnistForwardModule? : Except ValidationError Module := do
  let x := tensor "x" .f32 [2, 784]
  let w1 := tensor "w1" .f32 [784, 8]
  let b1 := tensor "b1" .f32 [8]
  let w2 := tensor "w2" .f32 [8, 10]
  let b2 := tensor "b2" .f32 [10]
  let hidden ← denseLayer "hidden" x w1 b1
  let activated ← reluActivation "relu" hidden.output
  let logits ← denseLayer "logits" activated.output w2 b2
  checkedModule "leanax_mnist_forward" "main" [x, w1, b1, w2, b2]
    (hidden.bindings ++ activated.bindings ++ logits.bindings)
    logits.output

end DSL
end LeanAX
