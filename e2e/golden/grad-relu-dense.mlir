module @leanax_grad_relu_dense {
  func.func @main(%x: tensor<2x784xf32>, %hidden_grad: tensor<2x8xf32>, %relu_mask: tensor<2x8xf32>) -> (tensor<784x8xf32>, tensor<8xf32>) {
    %pre_activation_grad = "stablehlo.multiply"(%hidden_grad, %relu_mask) : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    %x_t = "stablehlo.transpose"(%x) {permutation = [1, 0]} : (tensor<2x784xf32>) -> tensor<784x2xf32>
    %grad_w1 = "stablehlo.dot_general"(%x_t, %pre_activation_grad) {batching_dims = "[] x []"} : (tensor<784x2xf32>, tensor<2x8xf32>) -> tensor<784x8xf32>
    %pre_activation_grad_t = "stablehlo.transpose"(%pre_activation_grad) {permutation = [1, 0]} : (tensor<2x8xf32>) -> tensor<8x2xf32>
    %grad_b1_keepdim = "stablehlo.reduce"(%pre_activation_grad_t) {dimensions = "last"} : (tensor<8x2xf32>) -> tensor<8x1xf32>
    %grad_b1 = "stablehlo.reshape"(%grad_b1_keepdim) : (tensor<8x1xf32>) -> tensor<8xf32>
    return %grad_w1, %grad_b1 : tensor<784x8xf32>, tensor<8xf32>
  }
}
