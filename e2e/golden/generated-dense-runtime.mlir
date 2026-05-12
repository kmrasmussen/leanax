module {
  llvm.func @main() -> f32 {
    %x00 = llvm.mlir.constant(1.0 : f32) : f32
    %x01 = llvm.mlir.constant(-2.0 : f32) : f32
    %x10 = llvm.mlir.constant(0.5 : f32) : f32
    %x11 = llvm.mlir.constant(3.0 : f32) : f32
    %w00 = llvm.mlir.constant(2.0 : f32) : f32
    %w01 = llvm.mlir.constant(-1.0 : f32) : f32
    %w10 = llvm.mlir.constant(0.25 : f32) : f32
    %w11 = llvm.mlir.constant(1.5 : f32) : f32
    %b0 = llvm.mlir.constant(0.5 : f32) : f32
    %b1 = llvm.mlir.constant(-0.25 : f32) : f32
    %p000 = llvm.fmul %x00, %w00 : f32
    %p010 = llvm.fmul %x01, %w10 : f32
    %s00 = llvm.fadd %p000, %p010 : f32
    %y00 = llvm.fadd %s00, %b0 : f32
    %p001 = llvm.fmul %x00, %w01 : f32
    %p011 = llvm.fmul %x01, %w11 : f32
    %s01 = llvm.fadd %p001, %p011 : f32
    %y01 = llvm.fadd %s01, %b1 : f32
    %p100 = llvm.fmul %x10, %w00 : f32
    %p110 = llvm.fmul %x11, %w10 : f32
    %s10 = llvm.fadd %p100, %p110 : f32
    %y10 = llvm.fadd %s10, %b0 : f32
    %p101 = llvm.fmul %x10, %w01 : f32
    %p111 = llvm.fmul %x11, %w11 : f32
    %s11 = llvm.fadd %p101, %p111 : f32
    %y11 = llvm.fadd %s11, %b1 : f32
    %generated_dense_w0 = llvm.mlir.constant(1.0 : f32) : f32
    %generated_dense_p0 = llvm.fmul %y00, %generated_dense_w0 : f32
    %generated_dense_w1 = llvm.mlir.constant(2.0 : f32) : f32
    %generated_dense_p1 = llvm.fmul %y01, %generated_dense_w1 : f32
    %generated_dense_w2 = llvm.mlir.constant(3.0 : f32) : f32
    %generated_dense_p2 = llvm.fmul %y10, %generated_dense_w2 : f32
    %generated_dense_w3 = llvm.mlir.constant(4.0 : f32) : f32
    %generated_dense_p3 = llvm.fmul %y11, %generated_dense_w3 : f32
    %generated_dense_acc2 = llvm.fadd %generated_dense_p2, %generated_dense_p3 : f32
    %generated_dense_acc1 = llvm.fadd %generated_dense_p1, %generated_dense_acc2 : f32
    %generated_dense_acc0 = llvm.fadd %generated_dense_p0, %generated_dense_acc1 : f32
    llvm.return %generated_dense_acc0 : f32
  }
}
