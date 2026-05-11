module @leanax_cross_entropy_loss {
  func.func @main(%logits: tensor<2xf32>, %labels: tensor<2xf32>) -> tensor<f32> {
    %ce_exp = "stablehlo.exponential"(%logits) : (tensor<2xf32>) -> tensor<2xf32>
    %ce_denom = "stablehlo.reduce"(%ce_exp) {dimensions = "all"} : (tensor<2xf32>) -> tensor<f32>
    %ce_denom_batched = "stablehlo.broadcast_in_dim"(%ce_denom) : (tensor<f32>) -> tensor<2xf32>
    %ce_probs = "stablehlo.divide"(%ce_exp, %ce_denom_batched) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %ce_log_probs = "stablehlo.log"(%ce_probs) : (tensor<2xf32>) -> tensor<2xf32>
    %ce_weighted = "stablehlo.multiply"(%labels, %ce_log_probs) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %ce_sum = "stablehlo.reduce"(%ce_weighted) {dimensions = "all"} : (tensor<2xf32>) -> tensor<f32>
    %ce_neg_one = "stablehlo.constant"() {value = "-1.0"} : () -> tensor<f32>
    %ce_loss = "stablehlo.multiply"(%ce_sum, %ce_neg_one) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    return %ce_loss : tensor<f32>
  }
}
