module {
  llvm.func @main() -> f32 {
    %x00 = llvm.mlir.constant(1.0 : f32) : f32
    %x01 = llvm.mlir.constant(2.0 : f32) : f32
    %x02 = llvm.mlir.constant(3.0 : f32) : f32
    %x10 = llvm.mlir.constant(4.0 : f32) : f32
    %x11 = llvm.mlir.constant(5.0 : f32) : f32
    %x12 = llvm.mlir.constant(6.0 : f32) : f32
    %row0_s0 = llvm.fadd %x00, %x01 : f32
    %row0 = llvm.fadd %row0_s0, %x02 : f32
    %row1_s0 = llvm.fadd %x10, %x11 : f32
    %row1 = llvm.fadd %row1_s0, %x12 : f32
    %reduce_row_w0 = llvm.mlir.constant(1.0 : f32) : f32
    %reduce_row_p0 = llvm.fmul %row0, %reduce_row_w0 : f32
    %reduce_row_w1 = llvm.mlir.constant(2.0 : f32) : f32
    %reduce_row_p1 = llvm.fmul %row1, %reduce_row_w1 : f32
    %reduce_row_acc0 = llvm.fadd %reduce_row_p0, %reduce_row_p1 : f32
    llvm.return %reduce_row_acc0 : f32
  }
}
