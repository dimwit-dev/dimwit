package dimwit.autodiff

import dimwit.tensor.*
import dimwit.tensor.TensorOps.*

/** Convenience extension methods for operating on values with `ToFloatTensorTree` evidence.
  *
  * These are thin wrappers around `TensorTree` arithmetic methods that handle
  * wrapping/unwrapping automatically. Requires an explicit import:
  *
  * {{{
  * import dimwit.autodiff.FloatTensorTreeOps.*
  *
  * case class Params(w: Tensor1[A, Float], b: Tensor0[Float]) derives ToFloatTensorTree
  * val scaled = params **! Tensor0(0.1f)
  * }}}
  *
  * For chaining multiple operations, prefer working with `TensorTree` directly
  * (via `TensorTree.from(p)`) to avoid redundant Scala / JAX round-trips.
  */
object FloatTensorTreeOps:

  extension [P](p: P)(using ftt: ToFloatTensorTree[P])
    // Binary tree-vs-tree
    def ++(other: P): P = (TensorTree.from(p) ++ TensorTree.from(other)).toScala
    def --(other: P): P = (TensorTree.from(p) -- TensorTree.from(other)).toScala
    def **(other: P): P = (TensorTree.from(p) ** TensorTree.from(other)).toScala
    def `//`(other: P): P = (TensorTree.from(p) `//` TensorTree.from(other)).toScala

    // Broadcast tree-vs-scalar
    def ++!(scalar: Tensor0[Float]): P = TensorTree.from(p).++!(scalar).toScala
    def --!(scalar: Tensor0[Float]): P = TensorTree.from(p).--!(scalar).toScala
    def **!(scalar: Tensor0[Float]): P = TensorTree.from(p).**!(scalar).toScala
    def `//!`(scalar: Tensor0[Float]): P = TensorTree.from(p).`//!`(scalar).toScala

    // Utility
    def fillZeros: P = TensorTree.from(p).fillZeros.toScala
    def fillCopy(value: Float): P = TensorTree.from(p).fillCopy(value).toScala
