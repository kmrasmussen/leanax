module @leanax_mnist_cross_entropy {
  func.func @main(%logits: tensor<2x10xf32>, %labels: tensor<2x10xf32>) -> tensor<f32> {
    %mnist_ce_exp = "stablehlo.exponential"(%logits) : (tensor<2x10xf32>) -> tensor<2x10xf32>
    %mnist_ce_denom = "stablehlo.reduce"(%mnist_ce_exp) {dimensions = "last"} : (tensor<2x10xf32>) -> tensor<2x1xf32>
    %mnist_ce_denom_batched = "stablehlo.broadcast_in_dim"(%mnist_ce_denom) : (tensor<2x1xf32>) -> tensor<2x10xf32>
    %mnist_ce_probs = "stablehlo.divide"(%mnist_ce_exp, %mnist_ce_denom_batched) : (tensor<2x10xf32>, tensor<2x10xf32>) -> tensor<2x10xf32>
    %mnist_ce_log_probs = "stablehlo.log"(%mnist_ce_probs) : (tensor<2x10xf32>) -> tensor<2x10xf32>
    %mnist_ce_weighted = "stablehlo.multiply"(%labels, %mnist_ce_log_probs) : (tensor<2x10xf32>, tensor<2x10xf32>) -> tensor<2x10xf32>
    %mnist_ce_per_example = "stablehlo.reduce"(%mnist_ce_weighted) {dimensions = "last"} : (tensor<2x10xf32>) -> tensor<2x1xf32>
    %mnist_ce_sum = "stablehlo.reduce"(%mnist_ce_per_example) {dimensions = "all"} : (tensor<2x1xf32>) -> tensor<f32>
    %mnist_ce_neg_mean = "stablehlo.constant"() {value = "-0.5"} : () -> tensor<f32>
    %mnist_ce_loss = "stablehlo.multiply"(%mnist_ce_sum, %mnist_ce_neg_mean) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    return %mnist_ce_loss : tensor<f32>
  }
}
