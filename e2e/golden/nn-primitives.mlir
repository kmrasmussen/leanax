module @leanax_nn_primitives {
  func.func @main(%x: tensor<2x3xf32>, %bias: tensor<3xf32>) -> tensor<f32> {
    %scale = stablehlo.constant dense<2.0> : tensor<f32>
    %bias_batched = stablehlo.broadcast_in_dim %bias : tensor<2x3xf32>
    %shifted = stablehlo.add %x, %bias_batched : tensor<2x3xf32>
    %flat = stablehlo.reshape %shifted : tensor<6xf32>
    %restored = stablehlo.reshape %flat : tensor<2x3xf32>
    %transposed = stablehlo.transpose %restored, permutation = [1, 0] : tensor<3x2xf32>
    %scale_batched = stablehlo.broadcast_in_dim %scale : tensor<2x3xf32>
    %scaled = stablehlo.multiply %shifted, %scale_batched : tensor<2x3xf32>
    %loss = stablehlo.reduce %scaled, dimensions = all : tensor<f32>
    return %loss : tensor<f32>
  }
}
