module {
  llvm.func @main() -> f32 {
    %x0 = llvm.mlir.constant(1.0 : f32) : f32
    %x1 = llvm.mlir.constant(-2.0 : f32) : f32
    %x2 = llvm.mlir.constant(0.5 : f32) : f32
    %x3 = llvm.mlir.constant(3.0 : f32) : f32
    %w00 = llvm.mlir.constant(1.0 : f32) : f32
    %w10 = llvm.mlir.constant(2.0 : f32) : f32
    %w20 = llvm.mlir.constant(-1.0 : f32) : f32
    %w30 = llvm.mlir.constant(0.25 : f32) : f32
    %b0 = llvm.mlir.constant(-1.0 : f32) : f32
    %p00 = llvm.fmul %x0, %w00 : f32
    %p10 = llvm.fmul %x1, %w10 : f32
    %s10 = llvm.fadd %p00, %p10 : f32
    %p20 = llvm.fmul %x2, %w20 : f32
    %s20 = llvm.fadd %s10, %p20 : f32
    %p30 = llvm.fmul %x3, %w30 : f32
    %s30 = llvm.fadd %s20, %p30 : f32
    %y0 = llvm.fadd %s30, %b0 : f32
    %w01 = llvm.mlir.constant(-1.0 : f32) : f32
    %w11 = llvm.mlir.constant(0.0 : f32) : f32
    %w21 = llvm.mlir.constant(1.5 : f32) : f32
    %w31 = llvm.mlir.constant(-2.0 : f32) : f32
    %b1 = llvm.mlir.constant(0.5 : f32) : f32
    %p01 = llvm.fmul %x0, %w01 : f32
    %p11 = llvm.fmul %x1, %w11 : f32
    %s11 = llvm.fadd %p01, %p11 : f32
    %p21 = llvm.fmul %x2, %w21 : f32
    %s21 = llvm.fadd %s11, %p21 : f32
    %p31 = llvm.fmul %x3, %w31 : f32
    %s31 = llvm.fadd %s21, %p31 : f32
    %y1 = llvm.fadd %s31, %b1 : f32
    %w02 = llvm.mlir.constant(0.5 : f32) : f32
    %w12 = llvm.mlir.constant(-0.5 : f32) : f32
    %w22 = llvm.mlir.constant(2.0 : f32) : f32
    %w32 = llvm.mlir.constant(1.0 : f32) : f32
    %b2 = llvm.mlir.constant(2.0 : f32) : f32
    %p02 = llvm.fmul %x0, %w02 : f32
    %p12 = llvm.fmul %x1, %w12 : f32
    %s12 = llvm.fadd %p02, %p12 : f32
    %p22 = llvm.fmul %x2, %w22 : f32
    %s22 = llvm.fadd %s12, %p22 : f32
    %p32 = llvm.fmul %x3, %w32 : f32
    %s32 = llvm.fadd %s22, %p32 : f32
    %y2 = llvm.fadd %s32, %b2 : f32
    %sq0 = llvm.fmul %y0, %y0 : f32
    %sq1 = llvm.fmul %y1, %y1 : f32
    %acc1 = llvm.fadd %sq0, %sq1 : f32
    %sq2 = llvm.fmul %y2, %y2 : f32
    %acc2 = llvm.fadd %acc1, %sq2 : f32
    llvm.return %acc2 : f32
  }
}
