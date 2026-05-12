module {
  llvm.func @main() -> f32 {
    %reshape_target_v0 = llvm.mlir.constant(1.0 : f32) : f32
    %reshape_target_w0 = llvm.mlir.constant(1.0 : f32) : f32
    %reshape_target_p0 = llvm.fmul %reshape_target_v0, %reshape_target_w0 : f32
    %reshape_target_v1 = llvm.mlir.constant(2.0 : f32) : f32
    %reshape_target_w1 = llvm.mlir.constant(2.0 : f32) : f32
    %reshape_target_p1 = llvm.fmul %reshape_target_v1, %reshape_target_w1 : f32
    %reshape_target_v2 = llvm.mlir.constant(3.0 : f32) : f32
    %reshape_target_w2 = llvm.mlir.constant(3.0 : f32) : f32
    %reshape_target_p2 = llvm.fmul %reshape_target_v2, %reshape_target_w2 : f32
    %reshape_target_v3 = llvm.mlir.constant(4.0 : f32) : f32
    %reshape_target_w3 = llvm.mlir.constant(4.0 : f32) : f32
    %reshape_target_p3 = llvm.fmul %reshape_target_v3, %reshape_target_w3 : f32
    %reshape_target_v4 = llvm.mlir.constant(5.0 : f32) : f32
    %reshape_target_w4 = llvm.mlir.constant(5.0 : f32) : f32
    %reshape_target_p4 = llvm.fmul %reshape_target_v4, %reshape_target_w4 : f32
    %reshape_target_v5 = llvm.mlir.constant(6.0 : f32) : f32
    %reshape_target_w5 = llvm.mlir.constant(6.0 : f32) : f32
    %reshape_target_p5 = llvm.fmul %reshape_target_v5, %reshape_target_w5 : f32
    %reshape_target_acc4 = llvm.fadd %reshape_target_p4, %reshape_target_p5 : f32
    %reshape_target_acc3 = llvm.fadd %reshape_target_p3, %reshape_target_acc4 : f32
    %reshape_target_acc2 = llvm.fadd %reshape_target_p2, %reshape_target_acc3 : f32
    %reshape_target_acc1 = llvm.fadd %reshape_target_p1, %reshape_target_acc2 : f32
    %reshape_target_acc0 = llvm.fadd %reshape_target_p0, %reshape_target_acc1 : f32
    llvm.return %reshape_target_acc0 : f32
  }
}
