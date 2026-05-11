module @leanax_square_sum {
  func.func @main(%x: tensor<2x3xf32>) -> tensor<f32> {
    %squared = "stablehlo.multiply"(%x, %x) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %loss = "stablehlo.reduce"(%squared) {dimensions = "all"} : (tensor<2x3xf32>) -> tensor<f32>
    return %loss : tensor<f32>
  }
}
