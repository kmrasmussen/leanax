module @leanax_mnist_parameter_tree {
  func.func @main(%w1: tensor<784x8xf32>, %b1: tensor<8xf32>, %w2: tensor<8x10xf32>, %b2: tensor<10xf32>, %grad_w1: tensor<784x8xf32>, %grad_b1: tensor<8xf32>, %grad_w2: tensor<8x10xf32>, %grad_b2: tensor<10xf32>) -> (tensor<784x8xf32>, tensor<8xf32>, tensor<8x10xf32>, tensor<10xf32>) {
    %mnist_neg_lr = "stablehlo.constant"() {value = "-0.1"} : () -> tensor<f32>
    %mnist_neg_lr_w1 = "stablehlo.broadcast_in_dim"(%mnist_neg_lr) : (tensor<f32>) -> tensor<784x8xf32>
    %delta_w1 = "stablehlo.multiply"(%grad_w1, %mnist_neg_lr_w1) : (tensor<784x8xf32>, tensor<784x8xf32>) -> tensor<784x8xf32>
    %next_w1 = "stablehlo.add"(%w1, %delta_w1) : (tensor<784x8xf32>, tensor<784x8xf32>) -> tensor<784x8xf32>
    %mnist_neg_lr_b1 = "stablehlo.broadcast_in_dim"(%mnist_neg_lr) : (tensor<f32>) -> tensor<8xf32>
    %delta_b1 = "stablehlo.multiply"(%grad_b1, %mnist_neg_lr_b1) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %next_b1 = "stablehlo.add"(%b1, %delta_b1) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %mnist_neg_lr_w2 = "stablehlo.broadcast_in_dim"(%mnist_neg_lr) : (tensor<f32>) -> tensor<8x10xf32>
    %delta_w2 = "stablehlo.multiply"(%grad_w2, %mnist_neg_lr_w2) : (tensor<8x10xf32>, tensor<8x10xf32>) -> tensor<8x10xf32>
    %next_w2 = "stablehlo.add"(%w2, %delta_w2) : (tensor<8x10xf32>, tensor<8x10xf32>) -> tensor<8x10xf32>
    %mnist_neg_lr_b2 = "stablehlo.broadcast_in_dim"(%mnist_neg_lr) : (tensor<f32>) -> tensor<10xf32>
    %delta_b2 = "stablehlo.multiply"(%grad_b2, %mnist_neg_lr_b2) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    %next_b2 = "stablehlo.add"(%b2, %delta_b2) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    return %next_w1, %next_b1, %next_w2, %next_b2 : tensor<784x8xf32>, tensor<8xf32>, tensor<8x10xf32>, tensor<10xf32>
  }
}
