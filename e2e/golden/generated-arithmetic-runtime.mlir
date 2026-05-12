module {
  llvm.func @main() -> f32 {
    %a = llvm.mlir.constant(1.5 : f32) : f32
    %b = llvm.mlir.constant(2.25 : f32) : f32
    %sum = llvm.fadd %a, %b : f32
    %scale = llvm.mlir.constant(0.5 : f32) : f32
    %scaled = llvm.fmul %sum, %scale : f32
    %offset = llvm.mlir.constant(0.125 : f32) : f32
    %checksum = llvm.fadd %scaled, %offset : f32
    llvm.return %checksum : f32
  }
}
