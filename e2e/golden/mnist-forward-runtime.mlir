module {
  llvm.func @main() -> f32 {
    %x0 = llvm.mlir.constant(1.0 : f32) : f32
    %x1 = llvm.mlir.constant(-2.0 : f32) : f32
    %x2 = llvm.mlir.constant(0.5 : f32) : f32
    %x3 = llvm.mlir.constant(3.0 : f32) : f32
    %zero = llvm.mlir.constant(0.0 : f32) : f32
    %h0 = llvm.mlir.constant(-3.75 : f32) : f32
    %h1 = llvm.mlir.constant(-5.75 : f32) : f32
    %h2 = llvm.mlir.constant(7.5 : f32) : f32
    %h0_pos = llvm.fcmp "ogt" %h0, %zero : f32
    %a0 = llvm.select %h0_pos, %h0, %zero : i1, f32
    %h1_pos = llvm.fcmp "ogt" %h1, %zero : f32
    %a1 = llvm.select %h1_pos, %h1, %zero : i1, f32
    %h2_pos = llvm.fcmp "ogt" %h2, %zero : f32
    %a2 = llvm.select %h2_pos, %h2, %zero : i1, f32
    %w20 = llvm.mlir.constant(1.0 : f32) : f32
    %w30 = llvm.mlir.constant(0.5 : f32) : f32
    %w40 = llvm.mlir.constant(-0.5 : f32) : f32
    %b0 = llvm.mlir.constant(0.1 : f32) : f32
    %p20 = llvm.fmul %a0, %w20 : f32
    %p30 = llvm.fmul %a1, %w30 : f32
    %s0 = llvm.fadd %p20, %p30 : f32
    %p40 = llvm.fmul %a2, %w40 : f32
    %s1 = llvm.fadd %s0, %p40 : f32
    %logit0 = llvm.fadd %s1, %b0 : f32
    %w21 = llvm.mlir.constant(-1.0 : f32) : f32
    %w31 = llvm.mlir.constant(2.0 : f32) : f32
    %w41 = llvm.mlir.constant(1.0 : f32) : f32
    %b1 = llvm.mlir.constant(-0.2 : f32) : f32
    %p21 = llvm.fmul %a0, %w21 : f32
    %p31 = llvm.fmul %a1, %w31 : f32
    %s2 = llvm.fadd %p21, %p31 : f32
    %p41 = llvm.fmul %a2, %w41 : f32
    %s3 = llvm.fadd %s2, %p41 : f32
    %logit1 = llvm.fadd %s3, %b1 : f32
    %sq0 = llvm.fmul %logit0, %logit0 : f32
    %sq1 = llvm.fmul %logit1, %logit1 : f32
    %checksum = llvm.fadd %sq0, %sq1 : f32
    llvm.return %checksum : f32
  }
}
