module @leanax_grad_dense_loss {
  func.func @main(%x: tensor<1x2xf32>, %w: tensor<2x2xf32>, %b: tensor<2xf32>) -> tensor<2x2xf32> {
    %linear = "stablehlo.dot_general"(%x, %w) {batching_dims = "[] x []"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
    %bias = "stablehlo.broadcast_in_dim"(%b) : (tensor<2xf32>) -> tensor<1x2xf32>
    %residual = "stablehlo.add"(%linear, %bias) : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
    %two = "stablehlo.constant"() {value = "2.0"} : () -> tensor<f32>
    %two_batched = "stablehlo.broadcast_in_dim"(%two) : (tensor<f32>) -> tensor<1x2xf32>
    %grad_out = "stablehlo.multiply"(%residual, %two_batched) : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
    %x_t = "stablehlo.transpose"(%x) {permutation = [1, 0]} : (tensor<1x2xf32>) -> tensor<2x1xf32>
    %grad_w = "stablehlo.dot_general"(%x_t, %grad_out) {batching_dims = "[] x []"} : (tensor<2x1xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    return %grad_w : tensor<2x2xf32>
  }
}
