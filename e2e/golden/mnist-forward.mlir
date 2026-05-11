module @leanax_mnist_forward {
  func.func @main(%x: tensor<2x784xf32>, %w1: tensor<784x8xf32>, %b1: tensor<8xf32>, %w2: tensor<8x10xf32>, %b2: tensor<10xf32>) -> tensor<2x10xf32> {
    %hidden_linear = "stablehlo.dot_general"(%x, %w1) {batching_dims = "[] x []"} : (tensor<2x784xf32>, tensor<784x8xf32>) -> tensor<2x8xf32>
    %hidden_bias = "stablehlo.broadcast_in_dim"(%b1) : (tensor<8xf32>) -> tensor<2x8xf32>
    %hidden_out = "stablehlo.add"(%hidden_linear, %hidden_bias) : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    %relu_zero = "stablehlo.constant"() {value = "0.0"} : () -> tensor<f32>
    %relu_zero_batched = "stablehlo.broadcast_in_dim"(%relu_zero) : (tensor<f32>) -> tensor<2x8xf32>
    %relu_out = "stablehlo.maximum"(%hidden_out, %relu_zero_batched) : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    %logits_linear = "stablehlo.dot_general"(%relu_out, %w2) {batching_dims = "[] x []"} : (tensor<2x8xf32>, tensor<8x10xf32>) -> tensor<2x10xf32>
    %logits_bias = "stablehlo.broadcast_in_dim"(%b2) : (tensor<10xf32>) -> tensor<2x10xf32>
    %logits_out = "stablehlo.add"(%logits_linear, %logits_bias) : (tensor<2x10xf32>, tensor<2x10xf32>) -> tensor<2x10xf32>
    return %logits_out : tensor<2x10xf32>
  }
}
