import LeanAX.IR

namespace LeanAX

def affineRuntimeLLVM : String :=
  LeanAX.joinSep "\n" [
    "module {",
    "  llvm.func @main() -> f32 {",
    "    %x0 = llvm.mlir.constant(1.0 : f32) : f32",
    "    %b0 = llvm.mlir.constant(0.5 : f32) : f32",
    "    %sum0 = llvm.fadd %x0, %b0 : f32",
    "    %sq0 = llvm.fmul %sum0, %sum0 : f32",
    "    %x1 = llvm.mlir.constant(2.0 : f32) : f32",
    "    %b1 = llvm.mlir.constant(-1.0 : f32) : f32",
    "    %sum1 = llvm.fadd %x1, %b1 : f32",
    "    %sq1 = llvm.fmul %sum1, %sum1 : f32",
    "    %acc1 = llvm.fadd %sq0, %sq1 : f32",
    "    %x2 = llvm.mlir.constant(3.0 : f32) : f32",
    "    %b2 = llvm.mlir.constant(2.0 : f32) : f32",
    "    %sum2 = llvm.fadd %x2, %b2 : f32",
    "    %sq2 = llvm.fmul %sum2, %sum2 : f32",
    "    %acc2 = llvm.fadd %acc1, %sq2 : f32",
    "    %x3 = llvm.mlir.constant(4.0 : f32) : f32",
    "    %b3 = llvm.mlir.constant(1.0 : f32) : f32",
    "    %sum3 = llvm.fadd %x3, %b3 : f32",
    "    %sq3 = llvm.fmul %sum3, %sum3 : f32",
    "    %acc3 = llvm.fadd %acc2, %sq3 : f32",
    "    %x4 = llvm.mlir.constant(5.0 : f32) : f32",
    "    %b4 = llvm.mlir.constant(0.0 : f32) : f32",
    "    %sum4 = llvm.fadd %x4, %b4 : f32",
    "    %sq4 = llvm.fmul %sum4, %sum4 : f32",
    "    %acc4 = llvm.fadd %acc3, %sq4 : f32",
    "    %x5 = llvm.mlir.constant(6.0 : f32) : f32",
    "    %b5 = llvm.mlir.constant(-2.0 : f32) : f32",
    "    %sum5 = llvm.fadd %x5, %b5 : f32",
    "    %sq5 = llvm.fmul %sum5, %sum5 : f32",
    "    %acc5 = llvm.fadd %acc4, %sq5 : f32",
    "    llvm.return %acc5 : f32",
    "  }",
    "}",
    ""
  ]

def denseRuntimeLLVM : String :=
  LeanAX.joinSep "\n" [
    "module {",
    "  llvm.func @main() -> f32 {",
    "    %x0 = llvm.mlir.constant(1.0 : f32) : f32",
    "    %x1 = llvm.mlir.constant(-2.0 : f32) : f32",
    "    %x2 = llvm.mlir.constant(0.5 : f32) : f32",
    "    %x3 = llvm.mlir.constant(3.0 : f32) : f32",
    "    %w00 = llvm.mlir.constant(1.0 : f32) : f32",
    "    %w10 = llvm.mlir.constant(2.0 : f32) : f32",
    "    %w20 = llvm.mlir.constant(-1.0 : f32) : f32",
    "    %w30 = llvm.mlir.constant(0.25 : f32) : f32",
    "    %b0 = llvm.mlir.constant(-1.0 : f32) : f32",
    "    %p00 = llvm.fmul %x0, %w00 : f32",
    "    %p10 = llvm.fmul %x1, %w10 : f32",
    "    %s10 = llvm.fadd %p00, %p10 : f32",
    "    %p20 = llvm.fmul %x2, %w20 : f32",
    "    %s20 = llvm.fadd %s10, %p20 : f32",
    "    %p30 = llvm.fmul %x3, %w30 : f32",
    "    %s30 = llvm.fadd %s20, %p30 : f32",
    "    %y0 = llvm.fadd %s30, %b0 : f32",
    "    %w01 = llvm.mlir.constant(-1.0 : f32) : f32",
    "    %w11 = llvm.mlir.constant(0.0 : f32) : f32",
    "    %w21 = llvm.mlir.constant(1.5 : f32) : f32",
    "    %w31 = llvm.mlir.constant(-2.0 : f32) : f32",
    "    %b1 = llvm.mlir.constant(0.5 : f32) : f32",
    "    %p01 = llvm.fmul %x0, %w01 : f32",
    "    %p11 = llvm.fmul %x1, %w11 : f32",
    "    %s11 = llvm.fadd %p01, %p11 : f32",
    "    %p21 = llvm.fmul %x2, %w21 : f32",
    "    %s21 = llvm.fadd %s11, %p21 : f32",
    "    %p31 = llvm.fmul %x3, %w31 : f32",
    "    %s31 = llvm.fadd %s21, %p31 : f32",
    "    %y1 = llvm.fadd %s31, %b1 : f32",
    "    %w02 = llvm.mlir.constant(0.5 : f32) : f32",
    "    %w12 = llvm.mlir.constant(-0.5 : f32) : f32",
    "    %w22 = llvm.mlir.constant(2.0 : f32) : f32",
    "    %w32 = llvm.mlir.constant(1.0 : f32) : f32",
    "    %b2 = llvm.mlir.constant(2.0 : f32) : f32",
    "    %p02 = llvm.fmul %x0, %w02 : f32",
    "    %p12 = llvm.fmul %x1, %w12 : f32",
    "    %s12 = llvm.fadd %p02, %p12 : f32",
    "    %p22 = llvm.fmul %x2, %w22 : f32",
    "    %s22 = llvm.fadd %s12, %p22 : f32",
    "    %p32 = llvm.fmul %x3, %w32 : f32",
    "    %s32 = llvm.fadd %s22, %p32 : f32",
    "    %y2 = llvm.fadd %s32, %b2 : f32",
    "    %sq0 = llvm.fmul %y0, %y0 : f32",
    "    %sq1 = llvm.fmul %y1, %y1 : f32",
    "    %acc1 = llvm.fadd %sq0, %sq1 : f32",
    "    %sq2 = llvm.fmul %y2, %y2 : f32",
    "    %acc2 = llvm.fadd %acc1, %sq2 : f32",
    "    llvm.return %acc2 : f32",
    "  }",
    "}",
    ""
  ]

def runtimeLLVMByName (name : String) : Option String :=
  match name with
  | "affine-runtime" => some affineRuntimeLLVM
  | "dense-runtime" => some denseRuntimeLLVM
  | _ => none

def availableRuntimeCases : List String :=
  ["affine-runtime", "dense-runtime"]

end LeanAX
