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

def gradSoftmaxDenseModule? : Except ValidationError Module := do
  let hidden := tensor "hidden" .f32 [2, 8]
  let logits := tensor "logits" .f32 [2, 10]
  let labels := tensor "labels" .f32 [2, 10]
  let expLogits ← checkedExp "softmax_exp" logits
  let denom ← checkedReduceSumLastDim "softmax_denom" expLogits.result
  let denomBatched ← checkedBroadcastInDim "softmax_denom_batched" denom.result [2, 10]
  let probs ← checkedDivide "softmax_probs" expLogits.result denomBatched.result
  let negOne ← checkedConstant "neg_one" .f32 [] "-1.0"
  let negOneBatched ← checkedBroadcastInDim "neg_one_batched" negOne.result [2, 10]
  let negLabels ← checkedMultiply "neg_labels" labels negOneBatched.result
  let deltaUnscaled ← checkedAdd "logit_delta_unscaled" probs.result negLabels.result
  let meanScale ← checkedConstant "mean_scale" .f32 [] "0.5"
  let meanScaleBatched ← checkedBroadcastInDim "mean_scale_batched" meanScale.result [2, 10]
  let delta ← checkedMultiply "logit_delta" deltaUnscaled.result meanScaleBatched.result
  let hiddenT ← checkedTranspose "hidden_t" hidden [1, 0]
  let gradW2 ← checkedDotGeneral "grad_w2" hiddenT.result delta.result
  let deltaT ← checkedTranspose "logit_delta_t" delta.result [1, 0]
  let gradB2KeepDim ← checkedReduceSumLastDim "grad_b2_keepdim" deltaT.result
  let gradB2 ← checkedReshape "grad_b2" gradB2KeepDim.result [10]
  checkedModuleMulti "leanax_grad_softmax_dense" "main" [hidden, logits, labels]
    [
      expLogits,
      denom,
      denomBatched,
      probs,
      negOne,
      negOneBatched,
      negLabels,
      deltaUnscaled,
      meanScale,
      meanScaleBatched,
      delta,
      hiddenT,
      gradW2,
      deltaT,
      gradB2KeepDim,
      gradB2
    ]
    [gradW2.result, gradB2.result]

end LeanAX
