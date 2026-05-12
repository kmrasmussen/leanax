module @leanax_compare_select {
  func.func @main(%x: tensor<2x3xf32>, %threshold: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %pred = "stablehlo.compare"(%x, %threshold) {comparison_direction = "GT"} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi1>
    %out = "stablehlo.select"(%pred, %x, %threshold) : (tensor<2x3xi1>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %out : tensor<2x3xf32>
  }
}
