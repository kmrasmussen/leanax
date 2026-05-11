module @leanax_sgd_parameter_tree {
  func.func @main(%w: tensor<2x2xf32>, %b: tensor<2xf32>, %grad_w: tensor<2x2xf32>, %grad_b: tensor<2xf32>) -> (tensor<2x2xf32>, tensor<2xf32>) {
    %neg_lr = "stablehlo.constant"() {value = "-0.1"} : () -> tensor<f32>
    %neg_lr_w = "stablehlo.broadcast_in_dim"(%neg_lr) : (tensor<f32>) -> tensor<2x2xf32>
    %delta_w = "stablehlo.multiply"(%grad_w, %neg_lr_w) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %next_w = "stablehlo.add"(%w, %delta_w) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %neg_lr_b = "stablehlo.broadcast_in_dim"(%neg_lr) : (tensor<f32>) -> tensor<2xf32>
    %delta_b = "stablehlo.multiply"(%grad_b, %neg_lr_b) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %next_b = "stablehlo.add"(%b, %delta_b) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %next_w, %next_b : tensor<2x2xf32>, tensor<2xf32>
  }
}
