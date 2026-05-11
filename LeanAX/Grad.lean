import LeanAX.Build

namespace LeanAX

def squareSumModule? : Except ValidationError Module := do
  let x := tensor "x" .f32 [2, 3]
  let squared ← checkedMultiply "squared" x x
  let loss ← checkedReduceSum "loss" squared.result
  checkedModule "leanax_square_sum" "main" [x] [squared, loss] loss.result

def gradSquareSumModule? : Except ValidationError Module := do
  let x := tensor "x" .f32 [2, 3]
  let two ← checkedConstant "two" .f32 [] "2.0"
  let twoBatched ← checkedBroadcastInDim "two_batched" two.result [2, 3]
  let grad ← checkedMultiply "grad_x" x twoBatched.result
  checkedModule "leanax_grad_square_sum" "main" [x] [two, twoBatched, grad] grad.result

def gradDenseLossModule? : Except ValidationError Module := do
  let x := tensor "x" .f32 [1, 2]
  let w := tensor "w" .f32 [2, 2]
  let b := tensor "b" .f32 [2]
  let linear ← checkedDotGeneral "linear" x w
  let bias ← checkedBroadcastInDim "bias" b [1, 2]
  let residual ← checkedAdd "residual" linear.result bias.result
  let two ← checkedConstant "two" .f32 [] "2.0"
  let twoBatched ← checkedBroadcastInDim "two_batched" two.result [1, 2]
  let gradOut ← checkedMultiply "grad_out" residual.result twoBatched.result
  let xT ← checkedTranspose "x_t" x [1, 0]
  let gradW ← checkedDotGeneral "grad_w" xT.result gradOut.result
  checkedModule "leanax_grad_dense_loss" "main" [x, w, b]
    [linear, bias, residual, two, twoBatched, gradOut, xT, gradW]
    gradW.result

end LeanAX
