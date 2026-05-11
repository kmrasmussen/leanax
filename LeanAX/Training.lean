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

end LeanAX
