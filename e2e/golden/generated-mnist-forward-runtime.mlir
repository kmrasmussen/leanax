module {
  llvm.func @main() -> f32 {
    %x0 = llvm.mlir.constant(1.0 : f32) : f32
    %x1 = llvm.mlir.constant(-2.0 : f32) : f32
    %w100 = llvm.mlir.constant(0.5 : f32) : f32
    %w101 = llvm.mlir.constant(-0.25 : f32) : f32
    %w110 = llvm.mlir.constant(1.0 : f32) : f32
    %w111 = llvm.mlir.constant(0.75 : f32) : f32
    %b10 = llvm.mlir.constant(2.0 : f32) : f32
    %b11 = llvm.mlir.constant(1.0 : f32) : f32
    %w200 = llvm.mlir.constant(0.25 : f32) : f32
    %w201 = llvm.mlir.constant(-0.5 : f32) : f32
    %w210 = llvm.mlir.constant(1.5 : f32) : f32
    %w211 = llvm.mlir.constant(0.75 : f32) : f32
    %b20 = llvm.mlir.constant(0.05 : f32) : f32
    %b21 = llvm.mlir.constant(-0.1 : f32) : f32
    %zero = llvm.mlir.constant(0.0 : f32) : f32
    %h00 = llvm.fmul %x0, %w100 : f32
    %h01 = llvm.fmul %x1, %w110 : f32
    %h02 = llvm.fadd %h00, %h01 : f32
    %hidden_pre0 = llvm.fadd %h02, %b10 : f32
    %h10 = llvm.fmul %x0, %w101 : f32
    %h11 = llvm.fmul %x1, %w111 : f32
    %h12 = llvm.fadd %h10, %h11 : f32
    %hidden_pre1 = llvm.fadd %h12, %b11 : f32
    %hidden0_pos = llvm.fcmp "ogt" %hidden_pre0, %zero : f32
    %hidden1_pos = llvm.fcmp "ogt" %hidden_pre1, %zero : f32
    %hidden0 = llvm.select %hidden0_pos, %hidden_pre0, %zero : i1, f32
    %hidden1 = llvm.select %hidden1_pos, %hidden_pre1, %zero : i1, f32
    %l00 = llvm.fmul %hidden0, %w200 : f32
    %l01 = llvm.fmul %hidden1, %w210 : f32
    %l02 = llvm.fadd %l00, %l01 : f32
    %logit0 = llvm.fadd %l02, %b20 : f32
    %l10 = llvm.fmul %hidden0, %w201 : f32
    %l11 = llvm.fmul %hidden1, %w211 : f32
    %l12 = llvm.fadd %l10, %l11 : f32
    %logit1 = llvm.fadd %l12, %b21 : f32
    %generated_forward_w0 = llvm.mlir.constant(1.0 : f32) : f32
    %generated_forward_p0 = llvm.fmul %logit0, %generated_forward_w0 : f32
    %generated_forward_w1 = llvm.mlir.constant(2.0 : f32) : f32
    %generated_forward_p1 = llvm.fmul %logit1, %generated_forward_w1 : f32
    %generated_forward_acc0 = llvm.fadd %generated_forward_p0, %generated_forward_p1 : f32
    llvm.return %generated_forward_acc0 : f32
  }
}
