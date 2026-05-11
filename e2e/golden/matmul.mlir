module @leanax_matmul {
  func.func @main(%lhs: tensor<2x4xf32>, %rhs: tensor<4x3xf32>) -> tensor<2x3xf32> {
    %out = stablehlo.dot_general %lhs, %rhs, batching_dims = [] x [] : tensor<2x3xf32>
    return %out : tensor<2x3xf32>
  }
}
