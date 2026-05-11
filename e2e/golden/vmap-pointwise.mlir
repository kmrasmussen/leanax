module @leanax_vmap_pointwise {
  func.func @main(%x: tensor<4xf32>, %y: tensor<4xf32>) -> tensor<4xf32> {
    %sum = "stablehlo.add"(%x, %y) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %out = "stablehlo.multiply"(%sum, %sum) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %out : tensor<4xf32>
  }
}
