package dimwit.tensor.tensorops

import dimwit.linalg.LinearAlgebra
import dimwit.tensor.Axis
import dimwit.tensor.Label
import dimwit.tensor.Labels
import dimwit.tensor.Tensor
import dimwit.tensor.Tensor0
import dimwit.tensor.Tensor1
import dimwit.tensor.Tensor2
import dimwit.tensor.TensorOps.IsFloating
import dimwit.tensor.TensorOps.IsNumber

object LinearAlgebraOps:

  extension [L1: Label, L2: Label, V](t: Tensor2[L1, L2, V])

    /** return the diagonal of the tensor `t`
      * @see [[dimwit.linalg.LinearAlgebra.diagonal]] for details
      */
    def diagonal[LDiag: Label](diagAxis: Axis[LDiag]): Tensor1[LDiag, V] = LinearAlgebra.diagonal(t, diagAxis, 0)

    /** return the diagonal of the tensor `t`
      * @see [[dimwit.linalg.LinearAlgebra.diagonal]] for details
      */
    def diagonal[LDiag: Label](diagAxis: Axis[LDiag], offset: Int): Tensor1[LDiag, V] =
      LinearAlgebra.diagonal(t, diagAxis, offset)

  // ---------------------------------------------------------
  // IsNumber operations (IsFloat or IsInt)
  // ---------------------------------------------------------

  extension [L1: Label, L2: Label, V: IsNumber](t: Tensor2[L1, L2, V])

    /** Computes the trace of the tensor
      *
      * @see [[dimwit.linalg.LinearAlgebra.trace]] for details
      */
    def trace: Tensor0[V] = t.trace(0)

    /** Computes the trace. @see [[dimwit.linalg.LinearAlgebra.trace]] for details */
    def trace(offset: Int): Tensor0[V] = LinearAlgebra.trace(t, offset)

  extension [T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V])

    /** Computes the element wise L2 norm of the tensor t.
      *
      * @see [[dimwit.linalg.LinearAlgebra.norm]] for details
      */
    def norm: Tensor0[V] = LinearAlgebra.norm(t)

  extension [L: Label, V: IsFloating](v: Tensor1[L, V])

    /** Computes the vector norm of the 1-D tensor v.
      *
      * @see [[dimwit.linalg.LinearAlgebra.norm]] for details
      */
    def vectorNorm(normType: LinearAlgebra.VectorNormType): Tensor0[V] = LinearAlgebra.norm(v, normType)

  extension [LRow: Label, LCol: Label, V: IsFloating](m: Tensor2[LRow, LCol, V])

    /** Computes the matrix norm of the 2-D tensor m.
      *
      * @see [[dimwit.linalg.LinearAlgebra.norm]] for details
      */
    def matrixNorm(normType: LinearAlgebra.MatrixNormType): Tensor0[V] = LinearAlgebra.norm(m, normType)

  extension [LRow: Label, LCol: Label, V: IsFloating](t: Tensor2[LRow, LCol, V])
    /** computes the determinant of the 2-D tensor t
      * @see [[dimwit.linalg.LinearAlgebra.det]] for details
      */
    def det: Tensor0[V] = LinearAlgebra.det(t)

    /** @see [[dimwit.linalg.LinearAlgebra.inv]] for details
      */
    def inv: Tensor2[LCol, LRow, V] = LinearAlgebra.inv(t)
