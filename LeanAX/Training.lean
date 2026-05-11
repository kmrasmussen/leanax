import LeanAX.Build

namespace LeanAX

def linearTrainStepModule? : Except ValidationError Module := do
  let w := tensor "w" .f32 []
  let grad := tensor "grad" .f32 []
  let negLr ← checkedConstant "neg_lr" .f32 [] "-0.1"
  let delta ← checkedMultiply "delta" grad negLr.result
  let nextW ← checkedAdd "next_w" w delta.result
  checkedModule "leanax_linear_train_step" "main" [w, grad] [negLr, delta, nextW] nextW.result

end LeanAX
