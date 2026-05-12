module {
  llvm.func @main() -> f32 {
    %x00 = llvm.mlir.constant(1.0 : f32) : f32
    %x01 = llvm.mlir.constant(2.0 : f32) : f32
    %x02 = llvm.mlir.constant(3.0 : f32) : f32
    %x10 = llvm.mlir.constant(4.0 : f32) : f32
    %x11 = llvm.mlir.constant(5.0 : f32) : f32
    %x12 = llvm.mlir.constant(6.0 : f32) : f32
    %sum01 = llvm.fadd %x00, %x01 : f32
    %sum02 = llvm.fadd %sum01, %x02 : f32
    %sum03 = llvm.fadd %sum02, %x10 : f32
    %sum04 = llvm.fadd %sum03, %x11 : f32
    %checksum = llvm.fadd %sum04, %x12 : f32
    llvm.return %checksum : f32
  }
}
