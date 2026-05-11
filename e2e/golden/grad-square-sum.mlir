module @leanax_grad_square_sum {
  func.func @main(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %two = "stablehlo.constant"() {value = "2.0"} : () -> tensor<f32>
    %two_batched = "stablehlo.broadcast_in_dim"(%two) : (tensor<f32>) -> tensor<2x3xf32>
    %grad_x = "stablehlo.multiply"(%x, %two_batched) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %grad_x : tensor<2x3xf32>
  }
}
