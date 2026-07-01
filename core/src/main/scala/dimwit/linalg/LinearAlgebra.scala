package dimwit.linalg

import dimwit.jax.Jax
import dimwit.tensor.Axis
import dimwit.tensor.Label
import dimwit.tensor.Labels
import dimwit.tensor.ShapeTypeHelpers.AxesRemover
import dimwit.tensor.Tensor
import dimwit.tensor.Tensor0
import dimwit.tensor.Tensor1
import dimwit.tensor.Tensor2
import dimwit.tensor.TensorOps.IsFloating
import dimwit.tensor.TensorOps.IsNumber
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

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

  /** Computes the determinant of the tensor `t` along the specified axes (L1, L2)
    *
    * @param t The input tensor from which to compute the determinant.
    * @param axis1 The first axis along which to compute the determinant.
    * @param axis2 The second axis along which to compute the determinant.
    * @return A new tensor with the determinant computed, where the two specified axes are removed
    */
  def det[T <: Tuple: Labels, L1: Label, L2: Label, V: IsFloating](t: Tensor[T, V], axis1: Axis[L1], axis2: Axis[L2])(using
      ev: AxesRemover[T, (L1, L2)],
      labels: Labels[ev.RemainingAxes]
  ): Tensor[ev.RemainingAxes, V] =
    // JAX det only works on the last two axes (-2, -1). We must move the user's selected axes to the end.
    val moved = Jax.jnp.moveaxis(
      t.jaxValue,
      source = ev.indices.toPythonProxy,
      destination = Seq(-2, -1).toPythonProxy
    )
    Tensor(Jax.jnp.linalg.det(moved))

  /** Extracts the diagonal along the given two axes (with optional offset),
    * replacing them by a new 1D axis labeled L1.
    *
    * @param t The input tensor from which to extract the diagonal.
    * @param axis1 The first axis along which to extract the diagonal.
    * @param axis2 The second axis along which to extract the diagonal.
    * @param offset The offset of the diagonal from the main diagonal. Positive values indicate diagonals above the main diagonal, while negative values indicate diagonals below it.
    * @return A new tensor with the diagonal extracted, where the two specified axes are replaced by a new 1D axis labeled L1.
    */
  def diagonal[T <: Tuple, L1: Label, L2: Label, V](t: Tensor[T, V], axis1: Axis[L1], axis2: Axis[L2], offset: Int = 0)(using
      ev: AxesRemover[T, (L1, L2)],
      labels: Labels[ev.RemainingAxes]
  ): Tensor[ev.RemainingAxes *: L1 *: EmptyTuple, V] =
    Tensor(Jax.jnp.diagonal(t.jaxValue, offset = offset, axis1 = ev.indices(0), axis2 = ev.indices(1)))

  /** Computes the inverse of the tensor t along the last two axes.
    * The first axes of the tensors are preserved, while the last
    * two axes are replaced by their inverses.
    * The tensor must be square along the last two axes.
    *
    * @return a new tensor with the same shape as t, but with the last two axes replaced by their inverses.
    */
  def inv[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.linalg.inv(t.jaxValue))

  /** Computes the trace of the tensor `t` along the specified axes (L1, L2) with an optional offset.
    * The resulting tensor has the two specified axes removed, and the remaining axes are preserved.
    *
    * @param t The input tensor from which to compute the trace.
    * @param axis1 The first axis along which to compute the trace.
    * @param axis2 The second axis along which to compute the trace.
    * @param offset The offset of the diagonal from the main diagonal. Positive values indicate diagonals above the main diagonal, while negative values indicate diagonals below it.
    *
    * @return A new tensor with the trace computed, where the two specified axes are removed, and the remaining axes are preserved.
    */
  def trace[T <: Tuple, L1: Label, L2: Label, V: IsNumber](t: Tensor[T, V], axis1: Axis[L1], axis2: Axis[L2], offset: Int = 0)(using
      ev: AxesRemover[T, (L1, L2)],
      labels: Labels[ev.RemainingAxes]
  ): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.trace(t.jaxValue, offset = offset, axis1 = ev.indices(0), axis2 = ev.indices(1)))

  /** Computes the vector norm of the tensor `t` based on the specified `normType`.
    *
    * @param t The input tensor for which to compute the norm.
    * @param normType The type of norm to compute (L1, L2, Ord(p), or Inf).
    * @return A new 0-D tensor containing the computed norm of the input tensor.
    */
  def norm[L1: Label, V: IsFloating](t: Tensor1[L1, V], normType: VectorNormType): Tensor0[V] =
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
  def norm[L1: Label, L2: Label, V: IsFloating](t: Tensor2[L1, L2, V], normType: MatrixNormType): Tensor0[V] =
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
  def cholesky[L1: Label, L2: Label, V: IsFloating](t: Tensor2[L1, L2, V], upper: Boolean = false, symmetrizeInput: Boolean = true): Tensor2[L1, L2, V] =
    Tensor(Jax.jnp.linalg.cholesky(t.jaxValue, upper = upper, symmetrize_input = symmetrizeInput))

  /** Computes the QR factorization of the tensor `t`.
    *
    * @param t The input tensor to be factorized. It must be a 2D matrix.
    * @param mode The mode of the QR factorization (Reduced or Complete).
    * @return A tuple containing two tensors: the orthogonal matrix Q and the upper-triangular matrix R.
    *
    * @see [[https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.qr.html#jax.numpy.linalg.qr JAX documentation]] for more details on the underlying implementation.
    */
  def qr[L1: Label, L2: Label, V: IsFloating](t: Tensor2[L1, L2, V], mode: QRMode = QRMode.Reduced): (q: Tensor2[L1, L2, V], r: Tensor2[L1, L2, V]) =
    val qr = Jax.jnp.linalg.qr(
      t.jaxValue,
      mode = mode match
        case QRMode.Reduced  => "reduced"
        case QRMode.Complete => "complete"
    )
    (q = Tensor(qr.bracketAccess(0)), r = Tensor(qr.bracketAccess(1)))

  /** Computes the eigenvalues and eigenvectors of a symmetric matrix `t`.
    * @param t The input tensor representing a symmetric matrix.
    * @param upper If true, the upper-triangular part of the matrix is used.
    * @param symmetrizeInput If true, the input matrix is symmetrized before computation to ensure numerical stability.
    * @return A tuple containing two tensors: the eigenvalues and the corresponding eigenvectors of the input matrix.
    *
    * @see [[https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.eigh.html#jax.numpy.linalg.eigh JAX documentation]] for more details on the underlying implementation.
    */
  def eigh[L1: Label, L2: Label, V: IsFloating](t: Tensor2[L1, L2, V], upper: Boolean = false, symmetrizeInput: Boolean = true)
      : (eigenvalues: Tensor1[L1, V], eigenvectors: Tensor2[L1, L2, V]) =

    val ret = Jax.jnp.linalg.eigh(t.jaxValue, UPLO = if upper then "U" else "L", symmetrize_input = symmetrizeInput)
    val eigenvalues: Tensor1[L1, V] = Tensor(ret.bracketAccess(0))
    val eigenvectors: Tensor2[L1, L2, V] = Tensor(ret.bracketAccess(1))
    (eigenvalues = eigenvalues, eigenvectors = eigenvectors)

  /** Computes the singular value decomposition (SVD) of the tensor `t`.
    *
    * @param t The input tensor to be decomposed.
    * @param fullMatrices  If true, compute the full-sized U and Vh matrices; if false, compute the reduced-sized matrices.
    * @param hermitian If true, the input is a Hermitian matrix.
    * @return A tuple containing three tensors: the left singular vectors U, the singular values S, and the right singular vectors Vh.
    *
    * @see [[https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.svd.html#jax.numpy.linalg.svd JAX documentation]] for more details on the underlying implementation.
    */
  def svd[L1: Label, L2: Label, V: IsFloating](t: Tensor2[L1, L2, V], fullMatrices: Boolean = false, hermitian: Boolean = false)
      : (U: Tensor2[L1, L2, V], S: Tensor1[L1, V], Vh: Tensor2[L1, L2, V]) =

    val ret = Jax.jnp.linalg.svd(t.jaxValue, full_matrices = fullMatrices, hermitian = hermitian)
    val u: Tensor2[L1, L2, V] = Tensor(ret.bracketAccess(0))
    val s: Tensor1[L1, V] = Tensor(ret.bracketAccess(1))
    val vh: Tensor2[L1, L2, V] = Tensor(ret.bracketAccess(2))
    (U = u, S = s, Vh = vh)

  /** Solves the linear equation Ax = b for x, where A is a square matrix and b is a vector.
    *
    * @param a The input tensor representing the square matrix A.
    * @param b The input tensor representing the vector b.
    * @return A new tensor containing the solution vector x.
    *
    * @see [[https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.solve.html#jax.numpy.linalg.solve JAX documentation]] for more details on the underlying implementation.
    */
  def solve[L1: Label, L2: Label, V: IsFloating](a: Tensor2[L1, L2, V], b: Tensor1[L1, V]): Tensor1[L2, V] =
    Tensor(Jax.jnp.linalg.solve(a.jaxValue, b.jaxValue))
