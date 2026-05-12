import LeanAX.IR

namespace LeanAX

structure RuntimeLLVMProgram where
  body : List String
  result : String
  deriving Repr

def runtimeLLVMMain (body : List String) (result : String) : String :=
  LeanAX.joinSep "\n" (
    ["module {", "  llvm.func @main() -> f32 {"] ++
    body.map (fun line => "    " ++ line) ++
    ["    llvm.return %" ++ result ++ " : f32", "  }", "}", ""]
  )

def RuntimeLLVMProgram.render (program : RuntimeLLVMProgram) : String :=
  runtimeLLVMMain program.body program.result

def runtimeConstF32 (name : String) (value : String) : String :=
  "%" ++ name ++ " = llvm.mlir.constant(" ++ value ++ " : f32) : f32"

def runtimeBinaryF32 (result op lhs rhs : String) : String :=
  "%" ++ result ++ " = llvm." ++ op ++ " %" ++ lhs ++ ", %" ++ rhs ++ " : f32"

def generatedArithmeticRuntimeProgram : RuntimeLLVMProgram :=
  {
    body := [
      runtimeConstF32 "a" "1.5",
      runtimeConstF32 "b" "2.25",
      runtimeBinaryF32 "sum" "fadd" "a" "b",
      runtimeConstF32 "scale" "0.5",
      runtimeBinaryF32 "scaled" "fmul" "sum" "scale",
      runtimeConstF32 "offset" "0.125",
      runtimeBinaryF32 "checksum" "fadd" "scaled" "offset"
    ],
    result := "checksum"
  }

def generatedArithmeticRuntimeLLVM : String :=
  generatedArithmeticRuntimeProgram.render

def runtimeF32NatLit (value : Nat) : String :=
  toString value ++ ".0"

def runtimeWeightedChecksumAux
    (label : String)
    (index : Nat)
    (values : List String) : List String × String :=
  match values with
  | [] =>
      let zeroName := label ++ "_zero"
      ([runtimeConstF32 zeroName "0.0"], zeroName)
  | value :: rest =>
      let valueName := label ++ "_v" ++ toString index
      let weightName := label ++ "_w" ++ toString index
      let productName := label ++ "_p" ++ toString index
      let currentLines := [
        runtimeConstF32 valueName value,
        runtimeConstF32 weightName (runtimeF32NatLit (index + 1)),
        runtimeBinaryF32 productName "fmul" valueName weightName
      ]
      match rest with
      | [] => (currentLines, productName)
      | _ =>
          let tail := runtimeWeightedChecksumAux label (index + 1) rest
          let accName := label ++ "_acc" ++ toString index
          (currentLines ++ tail.fst ++ [runtimeBinaryF32 accName "fadd" productName tail.snd], accName)

def runtimeWeightedChecksumProgram (label : String) (values : List String) : RuntimeLLVMProgram :=
  let checksum := runtimeWeightedChecksumAux label 0 values
  { body := checksum.fst, result := checksum.snd }

def runtimeWeightedChecksumRefsAux
    (label : String)
    (index : Nat)
    (refs : List String) : List String × String :=
  match refs with
  | [] =>
      let zeroName := label ++ "_zero"
      ([runtimeConstF32 zeroName "0.0"], zeroName)
  | ref :: rest =>
      let weightName := label ++ "_w" ++ toString index
      let productName := label ++ "_p" ++ toString index
      let currentLines := [
        runtimeConstF32 weightName (runtimeF32NatLit (index + 1)),
        runtimeBinaryF32 productName "fmul" ref weightName
      ]
      match rest with
      | [] => (currentLines, productName)
      | _ =>
          let tail := runtimeWeightedChecksumRefsAux label (index + 1) rest
          let accName := label ++ "_acc" ++ toString index
          (currentLines ++ tail.fst ++ [runtimeBinaryF32 accName "fadd" productName tail.snd], accName)

def runtimeWeightedChecksumRefsProgram
    (body : List String)
    (label : String)
    (refs : List String) : RuntimeLLVMProgram :=
  let checksum := runtimeWeightedChecksumRefsAux label 0 refs
  { body := body ++ checksum.fst, result := checksum.snd }

def broadcastShapeRuntimeProgram : RuntimeLLVMProgram :=
  runtimeWeightedChecksumProgram
    "broadcast_target"
    ["1.0", "2.0", "3.0", "1.0", "2.0", "3.0"]

def reshapeShapeRuntimeProgram : RuntimeLLVMProgram :=
  runtimeWeightedChecksumProgram
    "reshape_target"
    ["1.0", "2.0", "3.0", "4.0", "5.0", "6.0"]

def transposeShapeRuntimeProgram : RuntimeLLVMProgram :=
  runtimeWeightedChecksumProgram
    "transpose_target"
    ["1.0", "4.0", "2.0", "5.0", "3.0", "6.0"]

def broadcastShapeRuntimeLLVM : String :=
  broadcastShapeRuntimeProgram.render

def reshapeShapeRuntimeLLVM : String :=
  reshapeShapeRuntimeProgram.render

def transposeShapeRuntimeLLVM : String :=
  transposeShapeRuntimeProgram.render

def reduceBaseConstants : List String :=
  [
    runtimeConstF32 "x00" "1.0",
    runtimeConstF32 "x01" "2.0",
    runtimeConstF32 "x02" "3.0",
    runtimeConstF32 "x10" "4.0",
    runtimeConstF32 "x11" "5.0",
    runtimeConstF32 "x12" "6.0"
  ]

def reduceRowSums : List String :=
  [
    runtimeBinaryF32 "row0_s0" "fadd" "x00" "x01",
    runtimeBinaryF32 "row0" "fadd" "row0_s0" "x02",
    runtimeBinaryF32 "row1_s0" "fadd" "x10" "x11",
    runtimeBinaryF32 "row1" "fadd" "row1_s0" "x12"
  ]

def reduceRowRuntimeProgram : RuntimeLLVMProgram :=
  runtimeWeightedChecksumRefsProgram
    (reduceBaseConstants ++ reduceRowSums)
    "reduce_row"
    ["row0", "row1"]

def reduceAllRuntimeProgram : RuntimeLLVMProgram :=
  {
    body := reduceBaseConstants ++ [
      runtimeBinaryF32 "sum01" "fadd" "x00" "x01",
      runtimeBinaryF32 "sum02" "fadd" "sum01" "x02",
      runtimeBinaryF32 "sum03" "fadd" "sum02" "x10",
      runtimeBinaryF32 "sum04" "fadd" "sum03" "x11",
      runtimeBinaryF32 "checksum" "fadd" "sum04" "x12"
    ],
    result := "checksum"
  }

def reduceKeepdimRuntimeProgram : RuntimeLLVMProgram :=
  runtimeWeightedChecksumRefsProgram
    (reduceBaseConstants ++ reduceRowSums)
    "reduce_keepdim"
    ["row0", "row0", "row0", "row1", "row1", "row1"]

def reduceRowRuntimeLLVM : String :=
  reduceRowRuntimeProgram.render

def reduceAllRuntimeLLVM : String :=
  reduceAllRuntimeProgram.render

def reduceKeepdimRuntimeLLVM : String :=
  reduceKeepdimRuntimeProgram.render

def generatedDenseRuntimeProgram : RuntimeLLVMProgram :=
  runtimeWeightedChecksumRefsProgram
    [
      runtimeConstF32 "x00" "1.0",
      runtimeConstF32 "x01" "-2.0",
      runtimeConstF32 "x10" "0.5",
      runtimeConstF32 "x11" "3.0",
      runtimeConstF32 "w00" "2.0",
      runtimeConstF32 "w01" "-1.0",
      runtimeConstF32 "w10" "0.25",
      runtimeConstF32 "w11" "1.5",
      runtimeConstF32 "b0" "0.5",
      runtimeConstF32 "b1" "-0.25",
      runtimeBinaryF32 "p000" "fmul" "x00" "w00",
      runtimeBinaryF32 "p010" "fmul" "x01" "w10",
      runtimeBinaryF32 "s00" "fadd" "p000" "p010",
      runtimeBinaryF32 "y00" "fadd" "s00" "b0",
      runtimeBinaryF32 "p001" "fmul" "x00" "w01",
      runtimeBinaryF32 "p011" "fmul" "x01" "w11",
      runtimeBinaryF32 "s01" "fadd" "p001" "p011",
      runtimeBinaryF32 "y01" "fadd" "s01" "b1",
      runtimeBinaryF32 "p100" "fmul" "x10" "w00",
      runtimeBinaryF32 "p110" "fmul" "x11" "w10",
      runtimeBinaryF32 "s10" "fadd" "p100" "p110",
      runtimeBinaryF32 "y10" "fadd" "s10" "b0",
      runtimeBinaryF32 "p101" "fmul" "x10" "w01",
      runtimeBinaryF32 "p111" "fmul" "x11" "w11",
      runtimeBinaryF32 "s11" "fadd" "p101" "p111",
      runtimeBinaryF32 "y11" "fadd" "s11" "b1"
    ]
    "generated_dense"
    ["y00", "y01", "y10", "y11"]

def generatedDenseRuntimeLLVM : String :=
  generatedDenseRuntimeProgram.render

def generatedMnistForwardRuntimeProgram : RuntimeLLVMProgram :=
  runtimeWeightedChecksumRefsProgram
    [
      runtimeConstF32 "x0" "1.0",
      runtimeConstF32 "x1" "-2.0",
      runtimeConstF32 "w100" "0.5",
      runtimeConstF32 "w101" "-0.25",
      runtimeConstF32 "w110" "1.0",
      runtimeConstF32 "w111" "0.75",
      runtimeConstF32 "b10" "2.0",
      runtimeConstF32 "b11" "1.0",
      runtimeConstF32 "w200" "0.25",
      runtimeConstF32 "w201" "-0.5",
      runtimeConstF32 "w210" "1.5",
      runtimeConstF32 "w211" "0.75",
      runtimeConstF32 "b20" "0.05",
      runtimeConstF32 "b21" "-0.1",
      runtimeConstF32 "zero" "0.0",
      runtimeBinaryF32 "h00" "fmul" "x0" "w100",
      runtimeBinaryF32 "h01" "fmul" "x1" "w110",
      runtimeBinaryF32 "h02" "fadd" "h00" "h01",
      runtimeBinaryF32 "hidden_pre0" "fadd" "h02" "b10",
      runtimeBinaryF32 "h10" "fmul" "x0" "w101",
      runtimeBinaryF32 "h11" "fmul" "x1" "w111",
      runtimeBinaryF32 "h12" "fadd" "h10" "h11",
      runtimeBinaryF32 "hidden_pre1" "fadd" "h12" "b11",
      "%hidden0_pos = llvm.fcmp \"ogt\" %hidden_pre0, %zero : f32",
      "%hidden1_pos = llvm.fcmp \"ogt\" %hidden_pre1, %zero : f32",
      "%hidden0 = llvm.select %hidden0_pos, %hidden_pre0, %zero : i1, f32",
      "%hidden1 = llvm.select %hidden1_pos, %hidden_pre1, %zero : i1, f32",
      runtimeBinaryF32 "l00" "fmul" "hidden0" "w200",
      runtimeBinaryF32 "l01" "fmul" "hidden1" "w210",
      runtimeBinaryF32 "l02" "fadd" "l00" "l01",
      runtimeBinaryF32 "logit0" "fadd" "l02" "b20",
      runtimeBinaryF32 "l10" "fmul" "hidden0" "w201",
      runtimeBinaryF32 "l11" "fmul" "hidden1" "w211",
      runtimeBinaryF32 "l12" "fadd" "l10" "l11",
      runtimeBinaryF32 "logit1" "fadd" "l12" "b21"
    ]
    "generated_forward"
    ["logit0", "logit1"]

def generatedMnistForwardRuntimeLLVM : String :=
  generatedMnistForwardRuntimeProgram.render

def generatedDerivedMaskTrainStepRuntimeProgram : RuntimeLLVMProgram :=
  {
    body := [
      runtimeConstF32 "x" "1.0",
      runtimeConstF32 "w1" "0.5",
      runtimeConstF32 "b1" "0.25",
      runtimeConstF32 "w20" "0.25",
      runtimeConstF32 "w21" "-0.5",
      runtimeConstF32 "b20" "0.05",
      runtimeConstF32 "b21" "-0.1",
      runtimeConstF32 "zero" "0.0",
      runtimeConstF32 "one" "1.0",
      runtimeConstF32 "neg_one" "-1.0",
      runtimeConstF32 "neg_lr" "-0.1",
      runtimeBinaryF32 "h0" "fmul" "x" "w1",
      runtimeBinaryF32 "hidden_pre" "fadd" "h0" "b1",
      "%mask_bool = llvm.fcmp \"ogt\" %hidden_pre, %zero : f32",
      "%hidden = llvm.select %mask_bool, %hidden_pre, %zero : i1, f32",
      "%mask = llvm.select %mask_bool, %one, %zero : i1, f32",
      runtimeBinaryF32 "l0" "fmul" "hidden" "w20",
      runtimeBinaryF32 "logit0" "fadd" "l0" "b20",
      runtimeBinaryF32 "l1" "fmul" "hidden" "w21",
      runtimeBinaryF32 "logit1" "fadd" "l1" "b21",
      "%exp0 = llvm.intr.exp(%logit0) : (f32) -> f32",
      "%exp1 = llvm.intr.exp(%logit1) : (f32) -> f32",
      runtimeBinaryF32 "denom" "fadd" "exp0" "exp1",
      runtimeBinaryF32 "prob0" "fdiv" "exp0" "denom",
      runtimeBinaryF32 "prob1" "fdiv" "exp1" "denom",
      "%log_prob1 = llvm.intr.log(%prob1) : (f32) -> f32",
      runtimeBinaryF32 "loss" "fmul" "log_prob1" "neg_one",
      runtimeBinaryF32 "delta0" "fadd" "prob0" "zero",
      runtimeBinaryF32 "delta1" "fadd" "prob1" "neg_one",
      runtimeBinaryF32 "grad_w20" "fmul" "hidden" "delta0",
      runtimeBinaryF32 "grad_w21" "fmul" "hidden" "delta1",
      runtimeBinaryF32 "hg0" "fmul" "delta0" "w20",
      runtimeBinaryF32 "hg1" "fmul" "delta1" "w21",
      runtimeBinaryF32 "hidden_grad" "fadd" "hg0" "hg1",
      runtimeBinaryF32 "pre_grad" "fmul" "hidden_grad" "mask",
      runtimeBinaryF32 "grad_w1" "fmul" "x" "pre_grad",
      runtimeBinaryF32 "dw1" "fmul" "grad_w1" "neg_lr",
      runtimeBinaryF32 "next_w1" "fadd" "w1" "dw1",
      runtimeBinaryF32 "db1" "fmul" "pre_grad" "neg_lr",
      runtimeBinaryF32 "next_b1" "fadd" "b1" "db1",
      runtimeBinaryF32 "dw20" "fmul" "grad_w20" "neg_lr",
      runtimeBinaryF32 "next_w20" "fadd" "w20" "dw20",
      runtimeBinaryF32 "dw21" "fmul" "grad_w21" "neg_lr",
      runtimeBinaryF32 "next_w21" "fadd" "w21" "dw21",
      runtimeBinaryF32 "db20" "fmul" "delta0" "neg_lr",
      runtimeBinaryF32 "next_b20" "fadd" "b20" "db20",
      runtimeBinaryF32 "db21" "fmul" "delta1" "neg_lr",
      runtimeBinaryF32 "next_b21" "fadd" "b21" "db21",
      runtimeBinaryF32 "sum0" "fadd" "loss" "next_w1",
      runtimeBinaryF32 "sum1" "fadd" "sum0" "next_b1",
      runtimeBinaryF32 "sum2" "fadd" "sum1" "next_w20",
      runtimeBinaryF32 "sum3" "fadd" "sum2" "next_w21",
      runtimeBinaryF32 "sum4" "fadd" "sum3" "next_b20",
      runtimeBinaryF32 "checksum" "fadd" "sum4" "next_b21"
    ],
    result := "checksum"
  }

def generatedDerivedMaskTrainStepRuntimeLLVM : String :=
  generatedDerivedMaskTrainStepRuntimeProgram.render

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

def mnistForwardRuntimeLLVM : String :=
  LeanAX.joinSep "\n" [
    "module {",
    "  llvm.func @main() -> f32 {",
    "    %x0 = llvm.mlir.constant(1.0 : f32) : f32",
    "    %x1 = llvm.mlir.constant(-2.0 : f32) : f32",
    "    %x2 = llvm.mlir.constant(0.5 : f32) : f32",
    "    %x3 = llvm.mlir.constant(3.0 : f32) : f32",
    "    %zero = llvm.mlir.constant(0.0 : f32) : f32",
    "    %h0 = llvm.mlir.constant(-3.75 : f32) : f32",
    "    %h1 = llvm.mlir.constant(-5.75 : f32) : f32",
    "    %h2 = llvm.mlir.constant(7.5 : f32) : f32",
    "    %h0_pos = llvm.fcmp \"ogt\" %h0, %zero : f32",
    "    %a0 = llvm.select %h0_pos, %h0, %zero : i1, f32",
    "    %h1_pos = llvm.fcmp \"ogt\" %h1, %zero : f32",
    "    %a1 = llvm.select %h1_pos, %h1, %zero : i1, f32",
    "    %h2_pos = llvm.fcmp \"ogt\" %h2, %zero : f32",
    "    %a2 = llvm.select %h2_pos, %h2, %zero : i1, f32",
    "    %w20 = llvm.mlir.constant(1.0 : f32) : f32",
    "    %w30 = llvm.mlir.constant(0.5 : f32) : f32",
    "    %w40 = llvm.mlir.constant(-0.5 : f32) : f32",
    "    %b0 = llvm.mlir.constant(0.1 : f32) : f32",
    "    %p20 = llvm.fmul %a0, %w20 : f32",
    "    %p30 = llvm.fmul %a1, %w30 : f32",
    "    %s0 = llvm.fadd %p20, %p30 : f32",
    "    %p40 = llvm.fmul %a2, %w40 : f32",
    "    %s1 = llvm.fadd %s0, %p40 : f32",
    "    %logit0 = llvm.fadd %s1, %b0 : f32",
    "    %w21 = llvm.mlir.constant(-1.0 : f32) : f32",
    "    %w31 = llvm.mlir.constant(2.0 : f32) : f32",
    "    %w41 = llvm.mlir.constant(1.0 : f32) : f32",
    "    %b1 = llvm.mlir.constant(-0.2 : f32) : f32",
    "    %p21 = llvm.fmul %a0, %w21 : f32",
    "    %p31 = llvm.fmul %a1, %w31 : f32",
    "    %s2 = llvm.fadd %p21, %p31 : f32",
    "    %p41 = llvm.fmul %a2, %w41 : f32",
    "    %s3 = llvm.fadd %s2, %p41 : f32",
    "    %logit1 = llvm.fadd %s3, %b1 : f32",
    "    %sq0 = llvm.fmul %logit0, %logit0 : f32",
    "    %sq1 = llvm.fmul %logit1, %logit1 : f32",
    "    %checksum = llvm.fadd %sq0, %sq1 : f32",
    "    llvm.return %checksum : f32",
    "  }",
    "}",
    ""
  ]

def softmaxLossRuntimeLLVM : String :=
  LeanAX.joinSep "\n" [
    "module {",
    "  llvm.func @main() -> f32 {",
    "    %logit0 = llvm.mlir.constant(1.0 : f32) : f32",
    "    %logit1 = llvm.mlir.constant(2.0 : f32) : f32",
    "    %exp0 = llvm.intr.exp(%logit0) : (f32) -> f32",
    "    %exp1 = llvm.intr.exp(%logit1) : (f32) -> f32",
    "    %denom = llvm.fadd %exp0, %exp1 : f32",
    "    %prob1 = llvm.fdiv %exp1, %denom : f32",
    "    %log_prob1 = llvm.intr.log(%prob1) : (f32) -> f32",
    "    %neg_one = llvm.mlir.constant(-1.0 : f32) : f32",
    "    %loss = llvm.fmul %log_prob1, %neg_one : f32",
    "    llvm.return %loss : f32",
    "  }",
    "}",
    ""
  ]

def tinyTrainStepRuntimeLLVM : String :=
  LeanAX.joinSep "\n" [
    "module {",
    "  llvm.func @main() -> f32 {",
    "    %x0 = llvm.mlir.constant(1.0 : f32) : f32",
    "    %x1 = llvm.mlir.constant(-2.0 : f32) : f32",
    "    %w100 = llvm.mlir.constant(0.5 : f32) : f32",
    "    %w101 = llvm.mlir.constant(-0.25 : f32) : f32",
    "    %w110 = llvm.mlir.constant(1.0 : f32) : f32",
    "    %w111 = llvm.mlir.constant(0.75 : f32) : f32",
    "    %b10 = llvm.mlir.constant(2.0 : f32) : f32",
    "    %b11 = llvm.mlir.constant(1.0 : f32) : f32",
    "    %w200 = llvm.mlir.constant(0.25 : f32) : f32",
    "    %w201 = llvm.mlir.constant(-0.5 : f32) : f32",
    "    %w210 = llvm.mlir.constant(1.5 : f32) : f32",
    "    %w211 = llvm.mlir.constant(0.75 : f32) : f32",
    "    %b20 = llvm.mlir.constant(0.05 : f32) : f32",
    "    %b21 = llvm.mlir.constant(-0.1 : f32) : f32",
    "    %zero = llvm.mlir.constant(0.0 : f32) : f32",
    "    %one = llvm.mlir.constant(1.0 : f32) : f32",
    "    %neg_one = llvm.mlir.constant(-1.0 : f32) : f32",
    "    %neg_lr = llvm.mlir.constant(-0.1 : f32) : f32",
    "    %h00 = llvm.fmul %x0, %w100 : f32",
    "    %h01 = llvm.fmul %x1, %w110 : f32",
    "    %h02 = llvm.fadd %h00, %h01 : f32",
    "    %hidden_pre0 = llvm.fadd %h02, %b10 : f32",
    "    %h10 = llvm.fmul %x0, %w101 : f32",
    "    %h11 = llvm.fmul %x1, %w111 : f32",
    "    %h12 = llvm.fadd %h10, %h11 : f32",
    "    %hidden_pre1 = llvm.fadd %h12, %b11 : f32",
    "    %mask0_bool = llvm.fcmp \"ogt\" %hidden_pre0, %zero : f32",
    "    %mask1_bool = llvm.fcmp \"ogt\" %hidden_pre1, %zero : f32",
    "    %hidden0 = llvm.select %mask0_bool, %hidden_pre0, %zero : i1, f32",
    "    %hidden1 = llvm.select %mask1_bool, %hidden_pre1, %zero : i1, f32",
    "    %mask0 = llvm.select %mask0_bool, %one, %zero : i1, f32",
    "    %mask1 = llvm.select %mask1_bool, %one, %zero : i1, f32",
    "    %l00 = llvm.fmul %hidden0, %w200 : f32",
    "    %l01 = llvm.fmul %hidden1, %w210 : f32",
    "    %l02 = llvm.fadd %l00, %l01 : f32",
    "    %logit0 = llvm.fadd %l02, %b20 : f32",
    "    %l10 = llvm.fmul %hidden0, %w201 : f32",
    "    %l11 = llvm.fmul %hidden1, %w211 : f32",
    "    %l12 = llvm.fadd %l10, %l11 : f32",
    "    %logit1 = llvm.fadd %l12, %b21 : f32",
    "    %exp0 = llvm.intr.exp(%logit0) : (f32) -> f32",
    "    %exp1 = llvm.intr.exp(%logit1) : (f32) -> f32",
    "    %denom = llvm.fadd %exp0, %exp1 : f32",
    "    %prob0 = llvm.fdiv %exp0, %denom : f32",
    "    %prob1 = llvm.fdiv %exp1, %denom : f32",
    "    %log_prob1 = llvm.intr.log(%prob1) : (f32) -> f32",
    "    %loss = llvm.fmul %log_prob1, %neg_one : f32",
    "    %delta0 = llvm.fadd %prob0, %zero : f32",
    "    %delta1 = llvm.fadd %prob1, %neg_one : f32",
    "    %grad_w200 = llvm.fmul %hidden0, %delta0 : f32",
    "    %grad_w201 = llvm.fmul %hidden0, %delta1 : f32",
    "    %grad_w210 = llvm.fmul %hidden1, %delta0 : f32",
    "    %grad_w211 = llvm.fmul %hidden1, %delta1 : f32",
    "    %hg00 = llvm.fmul %delta0, %w200 : f32",
    "    %hg01 = llvm.fmul %delta1, %w201 : f32",
    "    %hidden_grad0 = llvm.fadd %hg00, %hg01 : f32",
    "    %hg10 = llvm.fmul %delta0, %w210 : f32",
    "    %hg11 = llvm.fmul %delta1, %w211 : f32",
    "    %hidden_grad1 = llvm.fadd %hg10, %hg11 : f32",
    "    %pre_grad0 = llvm.fmul %hidden_grad0, %mask0 : f32",
    "    %pre_grad1 = llvm.fmul %hidden_grad1, %mask1 : f32",
    "    %grad_w100 = llvm.fmul %x0, %pre_grad0 : f32",
    "    %grad_w101 = llvm.fmul %x0, %pre_grad1 : f32",
    "    %grad_w110 = llvm.fmul %x1, %pre_grad0 : f32",
    "    %grad_w111 = llvm.fmul %x1, %pre_grad1 : f32",
    "    %dw100 = llvm.fmul %grad_w100, %neg_lr : f32",
    "    %dw101 = llvm.fmul %grad_w101, %neg_lr : f32",
    "    %dw110 = llvm.fmul %grad_w110, %neg_lr : f32",
    "    %dw111 = llvm.fmul %grad_w111, %neg_lr : f32",
    "    %next_w100 = llvm.fadd %w100, %dw100 : f32",
    "    %next_w101 = llvm.fadd %w101, %dw101 : f32",
    "    %next_w110 = llvm.fadd %w110, %dw110 : f32",
    "    %next_w111 = llvm.fadd %w111, %dw111 : f32",
    "    %db10 = llvm.fmul %pre_grad0, %neg_lr : f32",
    "    %db11 = llvm.fmul %pre_grad1, %neg_lr : f32",
    "    %next_b10 = llvm.fadd %b10, %db10 : f32",
    "    %next_b11 = llvm.fadd %b11, %db11 : f32",
    "    %dw200 = llvm.fmul %grad_w200, %neg_lr : f32",
    "    %dw201 = llvm.fmul %grad_w201, %neg_lr : f32",
    "    %dw210 = llvm.fmul %grad_w210, %neg_lr : f32",
    "    %dw211 = llvm.fmul %grad_w211, %neg_lr : f32",
    "    %next_w200 = llvm.fadd %w200, %dw200 : f32",
    "    %next_w201 = llvm.fadd %w201, %dw201 : f32",
    "    %next_w210 = llvm.fadd %w210, %dw210 : f32",
    "    %next_w211 = llvm.fadd %w211, %dw211 : f32",
    "    %db20 = llvm.fmul %delta0, %neg_lr : f32",
    "    %db21 = llvm.fmul %delta1, %neg_lr : f32",
    "    %next_b20 = llvm.fadd %b20, %db20 : f32",
    "    %next_b21 = llvm.fadd %b21, %db21 : f32",
    "    %sum0 = llvm.fadd %loss, %next_w100 : f32",
    "    %sum1 = llvm.fadd %sum0, %next_w101 : f32",
    "    %sum2 = llvm.fadd %sum1, %next_w110 : f32",
    "    %sum3 = llvm.fadd %sum2, %next_w111 : f32",
    "    %sum4 = llvm.fadd %sum3, %next_b10 : f32",
    "    %sum5 = llvm.fadd %sum4, %next_b11 : f32",
    "    %sum6 = llvm.fadd %sum5, %next_w200 : f32",
    "    %sum7 = llvm.fadd %sum6, %next_w201 : f32",
    "    %sum8 = llvm.fadd %sum7, %next_w210 : f32",
    "    %sum9 = llvm.fadd %sum8, %next_w211 : f32",
    "    %sum10 = llvm.fadd %sum9, %next_b20 : f32",
    "    %checksum = llvm.fadd %sum10, %next_b21 : f32",
    "    llvm.return %checksum : f32",
    "  }",
    "}",
    ""
  ]

structure RuntimeLLVMCase where
  name : String
  llvm : String
  deriving Repr

def runtimeLLVMCases : List RuntimeLLVMCase :=
  [
    { name := "affine-runtime", llvm := affineRuntimeLLVM },
    { name := "dense-runtime", llvm := denseRuntimeLLVM },
    { name := "mnist-forward-runtime", llvm := mnistForwardRuntimeLLVM },
    { name := "softmax-loss-runtime", llvm := softmaxLossRuntimeLLVM },
    { name := "tiny-train-step-runtime", llvm := tinyTrainStepRuntimeLLVM },
    { name := "generated-arithmetic-runtime", llvm := generatedArithmeticRuntimeLLVM },
    { name := "broadcast-shape-runtime", llvm := broadcastShapeRuntimeLLVM },
    { name := "reshape-shape-runtime", llvm := reshapeShapeRuntimeLLVM },
    { name := "transpose-shape-runtime", llvm := transposeShapeRuntimeLLVM },
    { name := "reduce-row-runtime", llvm := reduceRowRuntimeLLVM },
    { name := "reduce-all-runtime", llvm := reduceAllRuntimeLLVM },
    { name := "reduce-keepdim-runtime", llvm := reduceKeepdimRuntimeLLVM },
    { name := "generated-dense-runtime", llvm := generatedDenseRuntimeLLVM },
    { name := "generated-mnist-forward-runtime", llvm := generatedMnistForwardRuntimeLLVM },
    {
      name := "generated-derived-mask-train-step-runtime",
      llvm := generatedDerivedMaskTrainStepRuntimeLLVM
    }
  ]

def runtimeLLVMByName (name : String) : Option String :=
  (runtimeLLVMCases.find? (fun runtimeCase => runtimeCase.name == name)).map
    (fun runtimeCase => runtimeCase.llvm)

def availableRuntimeCases : List String :=
  runtimeLLVMCases.map (fun runtimeCase => runtimeCase.name)

end LeanAX
