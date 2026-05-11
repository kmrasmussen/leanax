/-!
This file is not intended to compile yet. It is an API sketch for the first
LeanAX discussion.
-/

namespace LeanAX

inductive DType where
  | f32
  | i32
  | pred

abbrev Shape := List Nat

-- In a real implementation this would probably be an expression node, not an
-- eager Lean value containing tensor data.
opaque Tensor : DType -> Shape -> Type

opaque Compiled : Type -> Type -> Type

opaque add :
  Tensor dtype shape ->
  Tensor dtype shape ->
  Tensor dtype shape

opaque mul :
  Tensor dtype shape ->
  Tensor dtype shape ->
  Tensor dtype shape

opaque matmul :
  Tensor .f32 [m, k] ->
  Tensor .f32 [k, n] ->
  Tensor .f32 [m, n]

opaque reduceSum :
  Tensor .f32 shape ->
  Tensor .f32 []

opaque jit :
  (a -> b) ->
  Compiled a b

opaque grad :
  (Tensor .f32 shape -> Tensor .f32 []) ->
  (Tensor .f32 shape -> Tensor .f32 shape)

opaque vmap :
  (Tensor dtype shape -> Tensor dtype outShape) ->
  (Tensor dtype (batch :: shape) -> Tensor dtype (batch :: outShape))

def linear
    (x : Tensor .f32 [128, 784])
    (w : Tensor .f32 [784, 10]) :
    Tensor .f32 [128, 10] :=
  matmul x w

def loss
    (x : Tensor .f32 [128, 784])
    (w : Tensor .f32 [784, 10]) :
    Tensor .f32 [] :=
  reduceSum (linear x w)

opaque input : Tensor .f32 [128, 784]

-- Desired meaning:
-- compileLossGrad lowers a gradient program to StableHLO/XLA.
def compileLossGrad := jit (grad (fun w => loss input w))

end LeanAX
