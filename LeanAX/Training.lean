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

def mnistTrainStepModule? : Except ValidationError Module := do
  let x := tensor "x" .f32 [2, 784]
  let labels := tensor "labels" .f32 [2, 10]
  let reluMask := tensor "relu_mask" .f32 [2, 8]
  let w1 := tensor "w1" .f32 [784, 8]
  let b1 := tensor "b1" .f32 [8]
  let w2 := tensor "w2" .f32 [8, 10]
  let b2 := tensor "b2" .f32 [10]

  let hiddenLinear ← checkedDotGeneral "hidden_linear" x w1
  let hiddenBias ← checkedBroadcastInDim "hidden_bias" b1 [2, 8]
  let hiddenPre ← checkedAdd "hidden_pre" hiddenLinear.result hiddenBias.result
  let zero ← checkedConstant "relu_zero" .f32 [] "0.0"
  let zeroBatched ← checkedBroadcastInDim "relu_zero_batched" zero.result [2, 8]
  let hidden ← checkedMaximum "hidden" hiddenPre.result zeroBatched.result
  let logitsLinear ← checkedDotGeneral "logits_linear" hidden.result w2
  let logitsBias ← checkedBroadcastInDim "logits_bias" b2 [2, 10]
  let logits ← checkedAdd "logits" logitsLinear.result logitsBias.result

  let expLogits ← checkedExp "softmax_exp" logits.result
  let denom ← checkedReduceSumLastDim "softmax_denom" expLogits.result
  let denomBatched ← checkedBroadcastInDim "softmax_denom_batched" denom.result [2, 10]
  let probs ← checkedDivide "softmax_probs" expLogits.result denomBatched.result
  let logProbs ← checkedLog "softmax_log_probs" probs.result
  let weighted ← checkedMultiply "loss_weighted" labels logProbs.result
  let perExample ← checkedReduceSumLastDim "loss_per_example" weighted.result
  let lossSum ← checkedReduceSum "loss_sum" perExample.result
  let negMean ← checkedConstant "loss_neg_mean" .f32 [] "-0.5"
  let loss ← checkedMultiply "loss" lossSum.result negMean.result

  let negOne ← checkedConstant "neg_one" .f32 [] "-1.0"
  let negOneBatched ← checkedBroadcastInDim "neg_one_batched" negOne.result [2, 10]
  let negLabels ← checkedMultiply "neg_labels" labels negOneBatched.result
  let deltaUnscaled ← checkedAdd "logit_delta_unscaled" probs.result negLabels.result
  let meanScale ← checkedConstant "mean_scale" .f32 [] "0.5"
  let meanScaleBatched ← checkedBroadcastInDim "mean_scale_batched" meanScale.result [2, 10]
  let delta ← checkedMultiply "logit_delta" deltaUnscaled.result meanScaleBatched.result
  let hiddenT ← checkedTranspose "hidden_t" hidden.result [1, 0]
  let gradW2 ← checkedDotGeneral "grad_w2" hiddenT.result delta.result
  let deltaT ← checkedTranspose "logit_delta_t" delta.result [1, 0]
  let gradB2KeepDim ← checkedReduceSumLastDim "grad_b2_keepdim" deltaT.result
  let gradB2 ← checkedReshape "grad_b2" gradB2KeepDim.result [10]
  let w2T ← checkedTranspose "w2_t" w2 [1, 0]
  let hiddenGrad ← checkedDotGeneral "hidden_grad" delta.result w2T.result
  let preActivationGrad ← checkedMultiply "pre_activation_grad" hiddenGrad.result reluMask
  let xT ← checkedTranspose "x_t" x [1, 0]
  let gradW1 ← checkedDotGeneral "grad_w1" xT.result preActivationGrad.result
  let preActivationGradT ← checkedTranspose "pre_activation_grad_t" preActivationGrad.result [1, 0]
  let gradB1KeepDim ← checkedReduceSumLastDim "grad_b1_keepdim" preActivationGradT.result
  let gradB1 ← checkedReshape "grad_b1" gradB1KeepDim.result [8]

  let negLr ← checkedConstant "mnist_train_neg_lr" .f32 [] "-0.2"
  let negLrW1 ← checkedBroadcastInDim "mnist_train_neg_lr_w1" negLr.result [784, 8]
  let deltaW1 ← checkedMultiply "delta_w1" gradW1.result negLrW1.result
  let nextW1 ← checkedAdd "next_w1" w1 deltaW1.result
  let negLrB1 ← checkedBroadcastInDim "mnist_train_neg_lr_b1" negLr.result [8]
  let deltaB1 ← checkedMultiply "delta_b1" gradB1.result negLrB1.result
  let nextB1 ← checkedAdd "next_b1" b1 deltaB1.result
  let negLrW2 ← checkedBroadcastInDim "mnist_train_neg_lr_w2" negLr.result [8, 10]
  let deltaW2 ← checkedMultiply "delta_w2" gradW2.result negLrW2.result
  let nextW2 ← checkedAdd "next_w2" w2 deltaW2.result
  let negLrB2 ← checkedBroadcastInDim "mnist_train_neg_lr_b2" negLr.result [10]
  let deltaB2 ← checkedMultiply "delta_b2" gradB2.result negLrB2.result
  let nextB2 ← checkedAdd "next_b2" b2 deltaB2.result

  checkedModuleMulti "leanax_mnist_train_step" "main"
    [x, labels, reluMask, w1, b1, w2, b2]
    [
      hiddenLinear,
      hiddenBias,
      hiddenPre,
      zero,
      zeroBatched,
      hidden,
      logitsLinear,
      logitsBias,
      logits,
      expLogits,
      denom,
      denomBatched,
      probs,
      logProbs,
      weighted,
      perExample,
      lossSum,
      negMean,
      loss,
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
      gradB2,
      w2T,
      hiddenGrad,
      preActivationGrad,
      xT,
      gradW1,
      preActivationGradT,
      gradB1KeepDim,
      gradB1,
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
    [nextW1.result, nextB1.result, nextW2.result, nextB2.result, loss.result]

end LeanAX
