module @leanax_relu_derived_mask {
  func.func @main(%hidden_pre: tensor<2x8xf32>) -> (tensor<2x8xf32>, tensor<2x8xf32>) {
    %relu_zero = "stablehlo.constant"() {value = "0.0"} : () -> tensor<f32>
    %relu_one = "stablehlo.constant"() {value = "1.0"} : () -> tensor<f32>
    %relu_zero_batched = "stablehlo.broadcast_in_dim"(%relu_zero) : (tensor<f32>) -> tensor<2x8xf32>
    %relu_one_batched = "stablehlo.broadcast_in_dim"(%relu_one) : (tensor<f32>) -> tensor<2x8xf32>
    %relu_positive = "stablehlo.compare"(%hidden_pre, %relu_zero_batched) {comparison_direction = "GT"} : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xi1>
    %relu_hidden = "stablehlo.select"(%relu_positive, %hidden_pre, %relu_zero_batched) : (tensor<2x8xi1>, tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    %relu_mask = "stablehlo.select"(%relu_positive, %relu_one_batched, %relu_zero_batched) : (tensor<2x8xi1>, tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    return %relu_hidden, %relu_mask : tensor<2x8xf32>, tensor<2x8xf32>
  }
}
