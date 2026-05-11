import LeanAX.Build

namespace LeanAX

def linearTrainStepModule? : Except ValidationError Module := do
  let w := tensor "w" .f32 []
  let grad := tensor "grad" .f32 []
  let negLr ← checkedConstant "neg_lr" .f32 [] "-0.1"
  let delta ← checkedMultiply "delta" grad negLr.result
  let nextW ← checkedAdd "next_w" w delta.result
  checkedModule "leanax_linear_train_step" "main" [w, grad] [negLr, delta, nextW] nextW.result

def parameterTreeStepModule? : Except ValidationError Module := do
  let w := tensor "w" .f32 [2, 2]
  let b := tensor "b" .f32 [2]
  let gradW := tensor "grad_w" .f32 [2, 2]
  let gradB := tensor "grad_b" .f32 [2]
  let negLr ← checkedConstant "neg_lr" .f32 [] "-0.1"
  let negLrW ← checkedBroadcastInDim "neg_lr_w" negLr.result [2, 2]
  let deltaW ← checkedMultiply "delta_w" gradW negLrW.result
  let nextW ← checkedAdd "next_w" w deltaW.result
  let negLrB ← checkedBroadcastInDim "neg_lr_b" negLr.result [2]
  let deltaB ← checkedMultiply "delta_b" gradB negLrB.result
  let nextB ← checkedAdd "next_b" b deltaB.result
  checkedModuleMulti "leanax_sgd_parameter_tree" "main" [w, b, gradW, gradB]
    [negLr, negLrW, deltaW, nextW, negLrB, deltaB, nextB]
    [nextW.result, nextB.result]

def mnistParameterTreeStepModule? : Except ValidationError Module := do
  let w1 := tensor "w1" .f32 [784, 8]
  let b1 := tensor "b1" .f32 [8]
  let w2 := tensor "w2" .f32 [8, 10]
  let b2 := tensor "b2" .f32 [10]
  let gradW1 := tensor "grad_w1" .f32 [784, 8]
  let gradB1 := tensor "grad_b1" .f32 [8]
  let gradW2 := tensor "grad_w2" .f32 [8, 10]
  let gradB2 := tensor "grad_b2" .f32 [10]
  let negLr ← checkedConstant "mnist_neg_lr" .f32 [] "-0.1"
  let negLrW1 ← checkedBroadcastInDim "mnist_neg_lr_w1" negLr.result [784, 8]
  let deltaW1 ← checkedMultiply "delta_w1" gradW1 negLrW1.result
  let nextW1 ← checkedAdd "next_w1" w1 deltaW1.result
  let negLrB1 ← checkedBroadcastInDim "mnist_neg_lr_b1" negLr.result [8]
  let deltaB1 ← checkedMultiply "delta_b1" gradB1 negLrB1.result
  let nextB1 ← checkedAdd "next_b1" b1 deltaB1.result
  let negLrW2 ← checkedBroadcastInDim "mnist_neg_lr_w2" negLr.result [8, 10]
  let deltaW2 ← checkedMultiply "delta_w2" gradW2 negLrW2.result
  let nextW2 ← checkedAdd "next_w2" w2 deltaW2.result
  let negLrB2 ← checkedBroadcastInDim "mnist_neg_lr_b2" negLr.result [10]
  let deltaB2 ← checkedMultiply "delta_b2" gradB2 negLrB2.result
  let nextB2 ← checkedAdd "next_b2" b2 deltaB2.result
  checkedModuleMulti "leanax_mnist_parameter_tree" "main"
    [w1, b1, w2, b2, gradW1, gradB1, gradW2, gradB2]
    [
      negLr,
      negLrW1,
      deltaW1,
      nextW1,
      negLrB1,
      deltaB1,
      nextB1,
      negLrW2,
      deltaW2,
      nextW2,
      negLrB2,
      deltaB2,
      nextB2
    ]
    [nextW1.result, nextB1.result, nextW2.result, nextB2.result]

end LeanAX
