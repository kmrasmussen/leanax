module @leanax_relu_forward {
  func.func @main(%x: tensor<2x4xf32>, %w: tensor<4x3xf32>, %b: tensor<3xf32>) -> tensor<2x3xf32> {
    %hidden_linear = "stablehlo.dot_general"(%x, %w) {batching_dims = "[] x []"} : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
    %hidden_bias = "stablehlo.broadcast_in_dim"(%b) : (tensor<3xf32>) -> tensor<2x3xf32>
    %hidden_out = "stablehlo.add"(%hidden_linear, %hidden_bias) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %relu_zero = "stablehlo.constant"() {value = "0.0"} : () -> tensor<f32>
    %relu_zero_batched = "stablehlo.broadcast_in_dim"(%relu_zero) : (tensor<f32>) -> tensor<2x3xf32>
    %relu_out = "stablehlo.maximum"(%hidden_out, %relu_zero_batched) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %relu_out : tensor<2x3xf32>
  }
}
