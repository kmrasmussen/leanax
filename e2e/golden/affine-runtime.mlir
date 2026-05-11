module {
  llvm.func @main() -> f32 {
    %x0 = llvm.mlir.constant(1.0 : f32) : f32
    %b0 = llvm.mlir.constant(0.5 : f32) : f32
    %sum0 = llvm.fadd %x0, %b0 : f32
    %sq0 = llvm.fmul %sum0, %sum0 : f32
    %x1 = llvm.mlir.constant(2.0 : f32) : f32
    %b1 = llvm.mlir.constant(-1.0 : f32) : f32
    %sum1 = llvm.fadd %x1, %b1 : f32
    %sq1 = llvm.fmul %sum1, %sum1 : f32
    %acc1 = llvm.fadd %sq0, %sq1 : f32
    %x2 = llvm.mlir.constant(3.0 : f32) : f32
    %b2 = llvm.mlir.constant(2.0 : f32) : f32
    %sum2 = llvm.fadd %x2, %b2 : f32
    %sq2 = llvm.fmul %sum2, %sum2 : f32
    %acc2 = llvm.fadd %acc1, %sq2 : f32
    %x3 = llvm.mlir.constant(4.0 : f32) : f32
    %b3 = llvm.mlir.constant(1.0 : f32) : f32
    %sum3 = llvm.fadd %x3, %b3 : f32
    %sq3 = llvm.fmul %sum3, %sum3 : f32
    %acc3 = llvm.fadd %acc2, %sq3 : f32
    %x4 = llvm.mlir.constant(5.0 : f32) : f32
    %b4 = llvm.mlir.constant(0.0 : f32) : f32
    %sum4 = llvm.fadd %x4, %b4 : f32
    %sq4 = llvm.fmul %sum4, %sum4 : f32
    %acc4 = llvm.fadd %acc3, %sq4 : f32
    %x5 = llvm.mlir.constant(6.0 : f32) : f32
    %b5 = llvm.mlir.constant(-2.0 : f32) : f32
    %sum5 = llvm.fadd %x5, %b5 : f32
    %sq5 = llvm.fmul %sum5, %sum5 : f32
    %acc5 = llvm.fadd %acc4, %sq5 : f32
    llvm.return %acc5 : f32
  }
}
