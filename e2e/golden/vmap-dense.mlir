module @leanax_vmap_dense {
  func.func @main(%x: tensor<2x4xf32>, %w: tensor<4x3xf32>, %b: tensor<3xf32>) -> tensor<2x3xf32> {
    %dense_linear = "stablehlo.dot_general"(%x, %w) {batching_dims = "[] x []"} : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
    %dense_bias = "stablehlo.broadcast_in_dim"(%b) : (tensor<3xf32>) -> tensor<2x3xf32>
    %dense_out = "stablehlo.add"(%dense_linear, %dense_bias) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %dense_out : tensor<2x3xf32>
  }
}
