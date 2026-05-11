module @leanax_grad_softmax_dense {
  func.func @main(%hidden: tensor<2x8xf32>, %logits: tensor<2x10xf32>, %labels: tensor<2x10xf32>) -> (tensor<8x10xf32>, tensor<10xf32>) {
    %softmax_exp = "stablehlo.exponential"(%logits) : (tensor<2x10xf32>) -> tensor<2x10xf32>
    %softmax_denom = "stablehlo.reduce"(%softmax_exp) {dimensions = "last"} : (tensor<2x10xf32>) -> tensor<2x1xf32>
    %softmax_denom_batched = "stablehlo.broadcast_in_dim"(%softmax_denom) : (tensor<2x1xf32>) -> tensor<2x10xf32>
    %softmax_probs = "stablehlo.divide"(%softmax_exp, %softmax_denom_batched) : (tensor<2x10xf32>, tensor<2x10xf32>) -> tensor<2x10xf32>
    %neg_one = "stablehlo.constant"() {value = "-1.0"} : () -> tensor<f32>
    %neg_one_batched = "stablehlo.broadcast_in_dim"(%neg_one) : (tensor<f32>) -> tensor<2x10xf32>
    %neg_labels = "stablehlo.multiply"(%labels, %neg_one_batched) : (tensor<2x10xf32>, tensor<2x10xf32>) -> tensor<2x10xf32>
    %logit_delta_unscaled = "stablehlo.add"(%softmax_probs, %neg_labels) : (tensor<2x10xf32>, tensor<2x10xf32>) -> tensor<2x10xf32>
    %mean_scale = "stablehlo.constant"() {value = "0.5"} : () -> tensor<f32>
    %mean_scale_batched = "stablehlo.broadcast_in_dim"(%mean_scale) : (tensor<f32>) -> tensor<2x10xf32>
    %logit_delta = "stablehlo.multiply"(%logit_delta_unscaled, %mean_scale_batched) : (tensor<2x10xf32>, tensor<2x10xf32>) -> tensor<2x10xf32>
    %hidden_t = "stablehlo.transpose"(%hidden) {permutation = [1, 0]} : (tensor<2x8xf32>) -> tensor<8x2xf32>
    %grad_w2 = "stablehlo.dot_general"(%hidden_t, %logit_delta) {batching_dims = "[] x []"} : (tensor<8x2xf32>, tensor<2x10xf32>) -> tensor<8x10xf32>
    %logit_delta_t = "stablehlo.transpose"(%logit_delta) {permutation = [1, 0]} : (tensor<2x10xf32>) -> tensor<10x2xf32>
    %grad_b2_keepdim = "stablehlo.reduce"(%logit_delta_t) {dimensions = "last"} : (tensor<10x2xf32>) -> tensor<10x1xf32>
    %grad_b2 = "stablehlo.reshape"(%grad_b2_keepdim) : (tensor<10x1xf32>) -> tensor<10xf32>
    return %grad_w2, %grad_b2 : tensor<8x10xf32>, tensor<10xf32>
  }
}
