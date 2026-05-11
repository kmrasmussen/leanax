module @leanax_affine {
  func.func @main(%x: tensor<2x3xf32>, %bias: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %sum = "stablehlo.add"(%x, %bias) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %out = "stablehlo.multiply"(%sum, %sum) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %out : tensor<2x3xf32>
  }
}
