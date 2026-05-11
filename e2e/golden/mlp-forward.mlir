module @leanax_mlp_forward {
  func.func @main(%x: tensor<2x4xf32>, %w1: tensor<4x3xf32>, %b1: tensor<3xf32>, %w2: tensor<3x2xf32>, %b2: tensor<2xf32>) -> tensor<2x2xf32> {
    %hidden_linear = "stablehlo.dot_general"(%x, %w1) {batching_dims = "[] x []"} : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
    %hidden_bias = "stablehlo.broadcast_in_dim"(%b1) : (tensor<3xf32>) -> tensor<2x3xf32>
    %hidden_out = "stablehlo.add"(%hidden_linear, %hidden_bias) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %activation_out = "stablehlo.multiply"(%hidden_out, %hidden_out) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %logits_linear = "stablehlo.dot_general"(%activation_out, %w2) {batching_dims = "[] x []"} : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
    %logits_bias = "stablehlo.broadcast_in_dim"(%b2) : (tensor<2xf32>) -> tensor<2x2xf32>
    %logits_out = "stablehlo.add"(%logits_linear, %logits_bias) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %logits_out : tensor<2x2xf32>
  }
}
