package dimwit.linalg

import dimwit.python.PyIndex.itemAt
import dimwit.jax.Jax
import dimwit.tensor.Axis
import dimwit.tensor.Label
import dimwit.tensor.Labels
import dimwit.tensor.Tensor
import dimwit.tensor.Tensor0
import dimwit.tensor.Tensor1
import dimwit.tensor.Tensor2
import dimwit.tensor.TensorOps.IsFloating
import dimwit.tensor.TensorOps.IsNumber
import me.shadaj.scalapy.py

/**  Common linear algebra operations.
  */
object LinearAlgebra:

  enum VectorNormType:
    case L1
    case L2
    case Ord(p: Double)
    case Inf

  enum MatrixNormType:
    case Frobenius
    case Nuclear
    case Spectral
    case One
    case Inf

  enum QRMode:
    case Reduced
    case Complete

  /** Computes the determinant of the tensor `t`
    *
    * @param t The input tensor from which to compute the determinant.
    * @return The determinant of the input tensor
    */
  def det[LRow: Label, LCol: Label, V: IsFloating](t: Tensor[(LRow, LCol), V]): Tensor0[V] =
    Tensor(Jax.jnp.linalg.det(t.jaxValue))

  /** Extracts the diagonal, with an optional offset,
    *
    * @param t The input tensor from which to extract the diagonal.
    * @param diagAxis A new axis Label, representing the axis of the output diagonal tensor.
    * @param offset The offset of the diagonal from the main diagonal. Positive values indicate diagonals above the main diagonal, while negative values indicate diagonals below it.
    * @return A new tensor containing the extracted diagonal elements of the input tensor.
    */
  def diagonal[LRow: Label, LCol: Label, LDiag: Label, V](t: Tensor2[LRow, LCol, V], diagAxis: Axis[LDiag], offset: Int = 0): Tensor1[LDiag, V] =
    Tensor(Jax.jnp.diagonal(t.jaxValue, offset = offset))

  /** Computes the inverse of the rank 2 tensor t
    * @return a new tensor, representing the inverse of t
    */
  def inv[LRow: Label, LCol: Label, V: IsFloating](t: Tensor2[LRow, LCol, V]): Tensor2[LCol, LRow, V] = Tensor(Jax.jnp.linalg.inv(t.jaxValue))

  /** Computes the trace of the tensor `t` with an optional offset.
    *
    * @param t The input tensor from which to compute the trace.
    * @param offset The offset of the diagonal from the main diagonal. Positive values indicate diagonals above the main diagonal, while negative values indicate diagonals below it.
    *
    * @return The trace of the input tensor
    */
  def trace[LRow: Label, LCol: Label, V: IsNumber](t: Tensor2[LRow, LCol, V], offset: Int = 0): Tensor0[V] =
    Tensor0(Jax.jnp.trace(t.jaxValue, offset = offset))

  /** Computes the vector norm of the tensor `t` based on the specified `normType`.
    *
    * @param t The input tensor for which to compute the norm.
    * @param normType The type of norm to compute (L1, L2, Ord(p), or Inf).
    * @return A new 0-D tensor containing the computed norm of the input tensor.
    */
  def norm[L: Label, V: IsFloating](t: Tensor1[L, V], normType: VectorNormType): Tensor0[V] =
    normType match
      case VectorNormType.L1     => Tensor0(Jax.jnp.linalg.norm(t.jaxValue, ord = 1))
      case VectorNormType.L2     => Tensor0(Jax.jnp.linalg.norm(t.jaxValue, ord = 2))
      case VectorNormType.Ord(p) => Tensor0(Jax.jnp.linalg.norm(t.jaxValue, ord = p))
      case VectorNormType.Inf    => Tensor0(Jax.jnp.linalg.norm(t.jaxValue, ord = Jax.jnp.inf))

  /** Computes the matrix norm of the tensor `t` based on the specified `normType`.
    *
    * @param t The input tensor for which to compute the norm.
    * @param normType The type of norm to compute (Frobenius, Nuclear, Spectral, One, or Inf).
    * @return A new 0-D tensor containing the computed norm of the input tensor.
    */
  def norm[LRow: Label, LCol: Label, V: IsFloating](t: Tensor2[LRow, LCol, V], normType: MatrixNormType): Tensor0[V] =
    normType match
      case MatrixNormType.Frobenius => Tensor0(Jax.jnp.linalg.norm(t.jaxValue, ord = "fro"))
      case MatrixNormType.Nuclear   => Tensor0(Jax.jnp.linalg.norm(t.jaxValue, ord = "nuc"))
      case MatrixNormType.Spectral  => Tensor0(Jax.jnp.linalg.norm(t.jaxValue, ord = 2))
      case MatrixNormType.One       => Tensor0(Jax.jnp.linalg.norm(t.jaxValue, ord = 1))
      case MatrixNormType.Inf       => Tensor0(Jax.jnp.linalg.norm(t.jaxValue, ord = Jax.jnp.inf))

  /** Computes the element-wise norm of the tensor `t` l2 norm along the last axis.
    * @param t The input tensor for which to compute the norm.
    *
    * @return A new 0-D tensor containing the computed norm of the input tensor.
    */
  def norm[T <: Tuple, V: IsFloating](t: Tensor[T, V]): Tensor0[V] =
    Tensor0(Jax.jnp.linalg.norm(t.jaxValue))

  /** Cholesky factorization.
    *
    * @param t The input tensor to be factorized. It must be a symmetric positive-definite matrix.
    * @param upper If true, the upper-triangular Cholesky factor is returned
    * @param symmetrizeInput If true, the input matrix is symmetrized before factorization to ensure numerical stability.
    * @return a triangular matrix representing the cholesky factor
    *
    * @see [[https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.cholesky.html#jax.numpy.linalg.cholesky JAX documentation]] for more details on the underlying implementation.
    */
  def cholesky[LRow: Label, LCol: Label, V: IsFloating](t: Tensor2[LRow, LCol, V], upper: Boolean = false, symmetrizeInput: Boolean = true): Tensor2[LRow, LCol, V] =
    Tensor(Jax.jnp.linalg.cholesky(t.jaxValue, upper = upper, symmetrize_input = symmetrizeInput))

  /** Computes the QR factorization of the tensor `t`.
    *
    * @param t The input tensor to be factorized. It must be a 2D matrix.
    * @param basisAxis An axis Label denoting the basis axis of the output Q and R matrices.
    * @param mode The mode of the QR factorization (Reduced or Complete).
    * @return A tuple containing two tensors: the orthogonal matrix Q and the upper-triangular matrix R.
    *
    * @see [[https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.qr.html#jax.numpy.linalg.qr JAX documentation]] for more details on the underlying implementation.
    */
  def qr[LRow: Label, LCol: Label, LBasis: Label, V: IsFloating](t: Tensor2[LRow, LCol, V], basisAxis: Axis[LBasis], mode: QRMode = QRMode.Reduced): (q: Tensor2[LRow, LBasis, V], r: Tensor2[LBasis, LCol, V]) =
    val qr = Jax.jnp.linalg.qr(
      t.jaxValue,
      mode = mode match
        case QRMode.Reduced  => "reduced"
        case QRMode.Complete => "complete"
    )
    (q = Tensor[(LRow, LBasis), V](qr.itemAt(0)), r = Tensor[(LBasis, LCol), V](qr.itemAt(1)))

  /** Computes the eigenvalues and eigenvectors of a symmetric matrix `t`.
    * @param t The input tensor representing a symmetric matrix.
    * @param eigAxis An axis Label denoting the axis of the output eigenvalues tensor.
    * @param spaceAxis An axis Label denoting the axis of the output eigenvectors tensor.
    * @param upper If true, the upper-triangular part of the matrix is used.
    * @param symmetrizeInput If true, the input matrix is symmetrized before computation to ensure numerical stability.
    * @return A tuple containing two tensors: the eigenvalues and the corresponding eigenvectors of the input matrix.
    *
    * @see [[https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.eigh.html#jax.numpy.linalg.eigh JAX documentation]] for more details on the underlying implementation.
    */
  def eigh[LRow: Label, LCol: Label, LEig: Label, LSpace: Label, V: IsFloating](t: Tensor2[LRow, LCol, V], eigAxis: Axis[LEig], spaceAxis: Axis[LSpace], upper: Boolean = false, symmetrizeInput: Boolean = true)
      : (eigenvalues: Tensor1[LEig, V], eigenvectors: Tensor2[LSpace, LEig, V]) =

    val ret = Jax.jnp.linalg.eigh(t.jaxValue, UPLO = if upper then "U" else "L", symmetrize_input = symmetrizeInput)
    val eigenvalues: Tensor1[LEig, V] = Tensor(ret.itemAt(0))
    val eigenvectors: Tensor2[LSpace, LEig, V] = Tensor(ret.itemAt(1))
    (eigenvalues = eigenvalues, eigenvectors = eigenvectors)

  /** Computes the singular value decomposition (SVD) of the tensor `t`.
    *
    * @param t The input tensor to be decomposed.
    * @param basisAxis An new axis Label denoting the basis axis of the output U and Vh matrices.
    * @param singularValuesAxis An new axis Label denoting the axis of the output singular values tensor.
    * @param fullMatrices  If true, compute the full-sized U and Vh matrices; if false, compute the reduced-sized matrices.
    * @param hermitian If true, the input is a Hermitian matrix.
    * @return A tuple containing three tensors: the left singular vectors U, the singular values S, and the right singular vectors Vh.
    *
    * @see [[https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.svd.html#jax.numpy.linalg.svd JAX documentation]] for more details on the underlying implementation.
    */
  def svd[LRow: Label, LCol: Label, LBasis: Label, LSing: Label, V: IsFloating](t: Tensor2[LRow, LCol, V], basisAxis: Axis[LBasis], singularValuesAxis: Axis[LSing], fullMatrices: Boolean = false, hermitian: Boolean = false)
      : (U: Tensor2[LRow, LBasis, V], S: Tensor1[LSing, V], Vh: Tensor2[LBasis, LCol, V]) =

    val ret = Jax.jnp.linalg.svd(t.jaxValue, full_matrices = fullMatrices, hermitian = hermitian)
    val u: Tensor2[LRow, LBasis, V] = Tensor(ret.itemAt(0))
    val s: Tensor1[LSing, V] = Tensor(ret.itemAt(1))
    val vh: Tensor2[LBasis, LCol, V] = Tensor(ret.itemAt(2))
    (U = u, S = s, Vh = vh)

  /** Solves the linear equation Ax = b for x, where A is a square matrix and b is a vector.
    *
    * @param a The input tensor representing the square matrix A.
    * @param b The input tensor representing the vector b.
    * @return A new tensor containing the solution vector x.
    *
    * @see [[https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.solve.html#jax.numpy.linalg.solve JAX documentation]] for more details on the underlying implementation.
    */
  def solve[LRow: Label, LCol: Label, V: IsFloating](a: Tensor2[LRow, LCol, V], b: Tensor1[LRow, V]): Tensor1[LCol, V] =
    Tensor(Jax.jnp.linalg.solve(a.jaxValue, b.jaxValue))
