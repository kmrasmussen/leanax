module {
  llvm.func @main() -> f32 {
    %logit0 = llvm.mlir.constant(1.0 : f32) : f32
    %logit1 = llvm.mlir.constant(2.0 : f32) : f32
    %exp0 = llvm.intr.exp(%logit0) : (f32) -> f32
    %exp1 = llvm.intr.exp(%logit1) : (f32) -> f32
    %denom = llvm.fadd %exp0, %exp1 : f32
    %prob1 = llvm.fdiv %exp1, %denom : f32
    %log_prob1 = llvm.intr.log(%prob1) : (f32) -> f32
    %neg_one = llvm.mlir.constant(-1.0 : f32) : f32
    %loss = llvm.fmul %log_prob1, %neg_one : f32
    llvm.return %loss : f32
  }
}
