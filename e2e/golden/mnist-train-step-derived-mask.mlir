module @leanax_mnist_train_step_derived_mask {
  func.func @main(%x: tensor<2x784xf32>, %labels: tensor<2x10xf32>, %w1: tensor<784x8xf32>, %b1: tensor<8xf32>, %w2: tensor<8x10xf32>, %b2: tensor<10xf32>) -> (tensor<784x8xf32>, tensor<8xf32>, tensor<8x10xf32>, tensor<10xf32>, tensor<f32>) {
    %hidden_linear = "stablehlo.dot_general"(%x, %w1) {batching_dims = "[] x []"} : (tensor<2x784xf32>, tensor<784x8xf32>) -> tensor<2x8xf32>
    %hidden_bias = "stablehlo.broadcast_in_dim"(%b1) : (tensor<8xf32>) -> tensor<2x8xf32>
    %hidden_pre = "stablehlo.add"(%hidden_linear, %hidden_bias) : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    %relu_zero = "stablehlo.constant"() {value = "0.0"} : () -> tensor<f32>
    %relu_one = "stablehlo.constant"() {value = "1.0"} : () -> tensor<f32>
    %relu_zero_batched = "stablehlo.broadcast_in_dim"(%relu_zero) : (tensor<f32>) -> tensor<2x8xf32>
    %relu_one_batched = "stablehlo.broadcast_in_dim"(%relu_one) : (tensor<f32>) -> tensor<2x8xf32>
    %relu_positive = "stablehlo.compare"(%hidden_pre, %relu_zero_batched) {comparison_direction = "GT"} : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xi1>
    %relu_hidden = "stablehlo.select"(%relu_positive, %hidden_pre, %relu_zero_batched) : (tensor<2x8xi1>, tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    %relu_mask = "stablehlo.select"(%relu_positive, %relu_one_batched, %relu_zero_batched) : (tensor<2x8xi1>, tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    %logits_linear = "stablehlo.dot_general"(%relu_hidden, %w2) {batching_dims = "[] x []"} : (tensor<2x8xf32>, tensor<8x10xf32>) -> tensor<2x10xf32>
    %logits_bias = "stablehlo.broadcast_in_dim"(%b2) : (tensor<10xf32>) -> tensor<2x10xf32>
    %logits = "stablehlo.add"(%logits_linear, %logits_bias) : (tensor<2x10xf32>, tensor<2x10xf32>) -> tensor<2x10xf32>
    %softmax_exp = "stablehlo.exponential"(%logits) : (tensor<2x10xf32>) -> tensor<2x10xf32>
    %softmax_denom = "stablehlo.reduce"(%softmax_exp) {dimensions = "last"} : (tensor<2x10xf32>) -> tensor<2x1xf32>
    %softmax_denom_batched = "stablehlo.broadcast_in_dim"(%softmax_denom) : (tensor<2x1xf32>) -> tensor<2x10xf32>
    %softmax_probs = "stablehlo.divide"(%softmax_exp, %softmax_denom_batched) : (tensor<2x10xf32>, tensor<2x10xf32>) -> tensor<2x10xf32>
    %softmax_log_probs = "stablehlo.log"(%softmax_probs) : (tensor<2x10xf32>) -> tensor<2x10xf32>
    %loss_weighted = "stablehlo.multiply"(%labels, %softmax_log_probs) : (tensor<2x10xf32>, tensor<2x10xf32>) -> tensor<2x10xf32>
    %loss_per_example = "stablehlo.reduce"(%loss_weighted) {dimensions = "last"} : (tensor<2x10xf32>) -> tensor<2x1xf32>
    %loss_sum = "stablehlo.reduce"(%loss_per_example) {dimensions = "all"} : (tensor<2x1xf32>) -> tensor<f32>
    %loss_neg_mean = "stablehlo.constant"() {value = "-0.5"} : () -> tensor<f32>
    %loss = "stablehlo.multiply"(%loss_sum, %loss_neg_mean) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %neg_one = "stablehlo.constant"() {value = "-1.0"} : () -> tensor<f32>
    %neg_one_batched = "stablehlo.broadcast_in_dim"(%neg_one) : (tensor<f32>) -> tensor<2x10xf32>
    %neg_labels = "stablehlo.multiply"(%labels, %neg_one_batched) : (tensor<2x10xf32>, tensor<2x10xf32>) -> tensor<2x10xf32>
    %logit_delta_unscaled = "stablehlo.add"(%softmax_probs, %neg_labels) : (tensor<2x10xf32>, tensor<2x10xf32>) -> tensor<2x10xf32>
    %mean_scale = "stablehlo.constant"() {value = "0.5"} : () -> tensor<f32>
    %mean_scale_batched = "stablehlo.broadcast_in_dim"(%mean_scale) : (tensor<f32>) -> tensor<2x10xf32>
    %logit_delta = "stablehlo.multiply"(%logit_delta_unscaled, %mean_scale_batched) : (tensor<2x10xf32>, tensor<2x10xf32>) -> tensor<2x10xf32>
    %hidden_t = "stablehlo.transpose"(%relu_hidden) {permutation = [1, 0]} : (tensor<2x8xf32>) -> tensor<8x2xf32>
    %grad_w2 = "stablehlo.dot_general"(%hidden_t, %logit_delta) {batching_dims = "[] x []"} : (tensor<8x2xf32>, tensor<2x10xf32>) -> tensor<8x10xf32>
    %logit_delta_t = "stablehlo.transpose"(%logit_delta) {permutation = [1, 0]} : (tensor<2x10xf32>) -> tensor<10x2xf32>
    %grad_b2_keepdim = "stablehlo.reduce"(%logit_delta_t) {dimensions = "last"} : (tensor<10x2xf32>) -> tensor<10x1xf32>
    %grad_b2 = "stablehlo.reshape"(%grad_b2_keepdim) : (tensor<10x1xf32>) -> tensor<10xf32>
    %w2_t = "stablehlo.transpose"(%w2) {permutation = [1, 0]} : (tensor<8x10xf32>) -> tensor<10x8xf32>
    %hidden_grad = "stablehlo.dot_general"(%logit_delta, %w2_t) {batching_dims = "[] x []"} : (tensor<2x10xf32>, tensor<10x8xf32>) -> tensor<2x8xf32>
    %pre_activation_grad = "stablehlo.multiply"(%hidden_grad, %relu_mask) : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    %x_t = "stablehlo.transpose"(%x) {permutation = [1, 0]} : (tensor<2x784xf32>) -> tensor<784x2xf32>
    %grad_w1 = "stablehlo.dot_general"(%x_t, %pre_activation_grad) {batching_dims = "[] x []"} : (tensor<784x2xf32>, tensor<2x8xf32>) -> tensor<784x8xf32>
    %pre_activation_grad_t = "stablehlo.transpose"(%pre_activation_grad) {permutation = [1, 0]} : (tensor<2x8xf32>) -> tensor<8x2xf32>
    %grad_b1_keepdim = "stablehlo.reduce"(%pre_activation_grad_t) {dimensions = "last"} : (tensor<8x2xf32>) -> tensor<8x1xf32>
    %grad_b1 = "stablehlo.reshape"(%grad_b1_keepdim) : (tensor<8x1xf32>) -> tensor<8xf32>
    %mnist_train_neg_lr = "stablehlo.constant"() {value = "-0.2"} : () -> tensor<f32>
    %mnist_train_neg_lr_w1 = "stablehlo.broadcast_in_dim"(%mnist_train_neg_lr) : (tensor<f32>) -> tensor<784x8xf32>
    %delta_w1 = "stablehlo.multiply"(%grad_w1, %mnist_train_neg_lr_w1) : (tensor<784x8xf32>, tensor<784x8xf32>) -> tensor<784x8xf32>
    %next_w1 = "stablehlo.add"(%w1, %delta_w1) : (tensor<784x8xf32>, tensor<784x8xf32>) -> tensor<784x8xf32>
    %mnist_train_neg_lr_b1 = "stablehlo.broadcast_in_dim"(%mnist_train_neg_lr) : (tensor<f32>) -> tensor<8xf32>
    %delta_b1 = "stablehlo.multiply"(%grad_b1, %mnist_train_neg_lr_b1) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %next_b1 = "stablehlo.add"(%b1, %delta_b1) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %mnist_train_neg_lr_w2 = "stablehlo.broadcast_in_dim"(%mnist_train_neg_lr) : (tensor<f32>) -> tensor<8x10xf32>
    %delta_w2 = "stablehlo.multiply"(%grad_w2, %mnist_train_neg_lr_w2) : (tensor<8x10xf32>, tensor<8x10xf32>) -> tensor<8x10xf32>
    %next_w2 = "stablehlo.add"(%w2, %delta_w2) : (tensor<8x10xf32>, tensor<8x10xf32>) -> tensor<8x10xf32>
    %mnist_train_neg_lr_b2 = "stablehlo.broadcast_in_dim"(%mnist_train_neg_lr) : (tensor<f32>) -> tensor<10xf32>
    %delta_b2 = "stablehlo.multiply"(%grad_b2, %mnist_train_neg_lr_b2) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    %next_b2 = "stablehlo.add"(%b2, %delta_b2) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    return %next_w1, %next_b1, %next_w2, %next_b2, %loss : tensor<784x8xf32>, tensor<8xf32>, tensor<8x10xf32>, tensor<10xf32>, tensor<f32>
  }
}
