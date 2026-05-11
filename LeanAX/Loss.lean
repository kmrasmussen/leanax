import LeanAX.DSL

namespace LeanAX
namespace DSL

def crossEntropyLoss (namePrefix : String) (logits : ValueRef) (labels : ValueRef) :
    Except ValidationError LayerResult := do
  requireTensorEq "cross_entropy logits/labels" logits.ty labels.ty
  let expLogits ← checkedExp (namePrefix ++ "_exp") logits
  let denom ← checkedReduceSum (namePrefix ++ "_denom") expLogits.result
  let denomBatched ← checkedBroadcastInDim (namePrefix ++ "_denom_batched") denom.result logits.ty.shape
  let probs ← checkedDivide (namePrefix ++ "_probs") expLogits.result denomBatched.result
  let logProbs ← checkedLog (namePrefix ++ "_log_probs") probs.result
  let weighted ← checkedMultiply (namePrefix ++ "_weighted") labels logProbs.result
  let sum ← checkedReduceSum (namePrefix ++ "_sum") weighted.result
  let negOne ← checkedConstant (namePrefix ++ "_neg_one") .f32 [] "-1.0"
  let loss ← checkedMultiply (namePrefix ++ "_loss") sum.result negOne.result
  pure {
    bindings := [
      expLogits,
      denom,
      denomBatched,
      probs,
      logProbs,
      weighted,
      sum,
      negOne,
      loss
    ],
    output := loss.result
  }

def crossEntropyLossModule? : Except ValidationError Module := do
  let logits := tensor "logits" .f32 [2]
  let labels := tensor "labels" .f32 [2]
  let loss ← crossEntropyLoss "ce" logits labels
  checkedModule "leanax_cross_entropy_loss" "main" [logits, labels] loss.bindings loss.output

end DSL
end LeanAX
