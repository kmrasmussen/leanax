module @leanax_linear_train_step {
  func.func @main(%w: tensor<f32>, %grad: tensor<f32>) -> tensor<f32> {
    %neg_lr = "stablehlo.constant"() {value = "-0.1"} : () -> tensor<f32>
    %delta = "stablehlo.multiply"(%grad, %neg_lr) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %next_w = "stablehlo.add"(%w, %delta) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    return %next_w : tensor<f32>
  }
}
