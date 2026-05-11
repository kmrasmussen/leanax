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

end LeanAX
